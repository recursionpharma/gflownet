import io
import pickle
import queue
import threading
import traceback
from pickle import Pickler, Unpickler, UnpicklingError

import torch
import torch.multiprocessing as mp

DO_PIN_BUFFERS = False  # So actually, we don't really need to pin buffers, but it's a good idea in some cases.
# The shared memory is already a good step.


class SharedPinnedBuffer:
    def __init__(self, size):
        self.size = size
        self.buffer = torch.empty(size, dtype=torch.uint8)
        self.buffer.share_memory_()
        self.lock = mp.Lock()
        self.do_unreg = False

        if DO_PIN_BUFFERS and not self.buffer.is_pinned():
            # Sometimes torch will create an already pinned (page aligned) buffer, so we don't need to
            # pin it again; doing so will raise a CUDA error
            cudart = torch.cuda.cudart()
            r = cudart.cudaHostRegister(self.buffer.data_ptr(), self.buffer.numel() * self.buffer.element_size(), 0)
            assert r == 0
            self.do_unreg = True  # But then we need to unregister it later
            assert self.buffer.is_pinned()
        assert self.buffer.is_shared()

    def __del__(self):
        if torch.utils.data.get_worker_info() is None:
            if self.do_unreg:
                cudart = torch.cuda.cudart()
                r = cudart.cudaHostUnregister(self.buffer.data_ptr())
                assert r == 0


class _BufferPicklerSentinel:
    pass


class BufferPickler(Pickler):
    def __init__(self, buf: SharedPinnedBuffer):
        self._f = io.BytesIO()
        super().__init__(self._f)
        self.buf = buf
        # The lock will be released by the consumer (BufferUnpickler) of this buffer once
        # the memory has been transferred to the device and copied
        self.buf.lock.acquire()
        self.buf_offset = 0

    def persistent_id(self, v):
        if not isinstance(v, torch.Tensor):
            return None
        numel = v.numel() * v.element_size()
        if self.buf_offset + numel > self.buf.size:
            raise RuntimeError(
                f"Tried to allocate {self.buf_offset + numel} bytes in a buffer of size {self.buf.size}. "
                "Consider increasing cfg.mp_buffer_size"
            )
        start = self.buf_offset
        shape = tuple(v.shape)
        if v.ndim > 0 and v.stride(-1) != 1 or not v.is_contiguous():
            v = v.contiguous().reshape(-1)
        if v.ndim > 0 and v.stride(-1) != 1:
            # We're still not contiguous, this unfortunately happens occasionally, e.g.:
            # x = torch.arange(10).reshape((10, 1))
            # y = x.T[::2].T
            # y.stride(), y.is_contiguous(), y.contiguous().stride()
            # -> (1, 2), True, (1, 2)
            v = v.flatten() + 0
            # I don't know if this comes from my misunderstanding of strides or if it's a bug in torch
            # but either way torch will refuse to view this tensor as a uint8 tensor, so we have to + 0
            # to force torch to materialize it into a new tensor (it may otherwise be lazy and not materialize)
        if numel > 0:
            self.buf.buffer[start : start + numel] = v.flatten().view(torch.uint8)
        self.buf_offset += numel
        self.buf_offset += (8 - self.buf_offset % 8) % 8  # align to 8 bytes
        return (_BufferPicklerSentinel, (start, shape, v.dtype))

    def dumps(self, obj):
        self.dump(obj)
        return (self._f.getvalue(), self.buf_offset)


class BufferUnpickler(Unpickler):
    def __init__(self, buf: SharedPinnedBuffer, data, device):
        self._f, total_size = io.BytesIO(data[0]), data[1]
        super().__init__(self._f)
        self.buf = buf
        self.target_buf = buf.buffer[:total_size].to(device) + 0
        # Why the `+ 0`? Unfortunately, we have no way to know exactly when the consumer of the object we're
        # unpickling will be done using the buffer underlying the tensor, so we have to create a copy.
        # If we don't and another consumer starts using the buffer, and this consumer transfers this pinned
        # buffer to the GPU, the first consumer's tensors will be corrupted, because (depending on the CUDA
        # memory manager) the pinned buffer will transfer to the same GPU location.
        # Hopefully, especially if the target device is the GPU, the copy will be fast and/or async.
        # Note that this could be fixed by using one buffer for each worker, but that would be significantly
        # more memory usage.

    def load_tensor(self, offset, shape, dtype):
        numel = prod(shape) * dtype.itemsize
        tensor: torch.Tensor = self.target_buf[offset : offset + numel].view(dtype).view(shape)
        return tensor

    def persistent_load(self, pid):
        if isinstance(pid, tuple):
            sentinel, (offset, shape, dtype) = pid
            if sentinel is _BufferPicklerSentinel:
                return self.load_tensor(offset, shape, dtype)
        return UnpicklingError("Invalid persistent id")

    def load(self):
        r = super().load()
        # We're done with this buffer, release it for the next consumer
        self.buf.lock.release()
        return r


def prod(ns):
    p = 1
    for i in ns:
        p *= i
    return p


class MPObjectPlaceholder:
    """This class can be used for example as a model or dataset placeholder
    in a worker process, and translates calls to the object-placeholder into
    queries for the main process to execute on the real object."""

    def __init__(self, in_queues, out_queues, pickle_messages=False, shared_buffer_size=None):
        self.qs = in_queues, out_queues
        self.device = torch.device("cpu")
        self.pickle_messages = pickle_messages
        self._is_init = False
        self.shared_buffer_size = shared_buffer_size
        if shared_buffer_size:
            self._buffer_to_main = SharedPinnedBuffer(shared_buffer_size)
            self._buffer_from_main = SharedPinnedBuffer(shared_buffer_size)

    def _check_init(self):
        if self._is_init:
            return
        info = torch.utils.data.get_worker_info()
        if info is None:
            self.in_queue = self.qs[0][-1]
            self.out_queue = self.qs[1][-1]
        else:
            self.in_queue = self.qs[0][info.id]
            self.out_queue = self.qs[1][info.id]
        self._is_init = True

    def encode(self, m):
        if self.shared_buffer_size:
            return BufferPickler(self._buffer_to_main).dumps(m)
        if self.pickle_messages:
            return pickle.dumps(m)
        return m

    def decode(self, m):
        if self.shared_buffer_size:
            m = BufferUnpickler(self._buffer_from_main, m, self.device).load()
        if self.pickle_messages:
            m = pickle.loads(m)
        if isinstance(m, Exception):
            print("Received exception from main process, reraising.")
            raise m
        return m

    def __getattr__(self, name):
        def method_wrapper(*a, **kw):
            self._check_init()
            self.in_queue.put(self.encode((name, a, kw)))
            return self.decode(self.out_queue.get())

        return method_wrapper

    def __call__(self, *a, **kw):
        self._check_init()
        self.in_queue.put(self.encode(("__call__", a, kw)))
        return self.decode(self.out_queue.get())

    def __len__(self):
        self._check_init()
        self.in_queue.put(("__len__", (), {}))
        return self.out_queue.get()


class MPObjectProxy:
    """This class maintains a reference to some object and
    creates a `placeholder` attribute which can be safely passed to
    multiprocessing DataLoader workers.

    The placeholders in each process send messages accross multiprocessing
    queues which are received by this proxy instance. The proxy instance then
    runs the calls on our object and sends the return value back to the worker.

    Starts its own (daemon) thread.
    Always passes CPU tensors between processes.
    """

    def __init__(self, obj, num_workers: int, cast_types: tuple, pickle_messages: bool = False, sb_size=None):
        """Construct a multiprocessing object proxy.

        Parameters
        ----------
        obj: any python object to be proxied (typically a torch.nn.Module or ReplayBuffer)
            Lives in the main process to which method calls are passed
        num_workers: int
            Number of DataLoader workers
        cast_types: tuple
            Types that will be cast to cuda when received as arguments of method calls.
            torch.Tensor is cast by default.
        pickle_messages: bool
            If True, pickle messages sent between processes. This reduces load on shared
            memory, but increases load on CPU. It is recommended to activate this flag if
            encountering "Too many open files"-type errors.
        sb_size: Optional[int]
            shared buffer size
        """
        self.in_queues = [mp.Queue() for i in range(num_workers + 1)]  # type: ignore
        self.out_queues = [mp.Queue() for i in range(num_workers + 1)]  # type: ignore
        self.pickle_messages = pickle_messages
        self.use_shared_buffer = bool(sb_size)
        self.placeholder = MPObjectPlaceholder(self.in_queues, self.out_queues, pickle_messages, sb_size)
        self.obj = obj
        if hasattr(obj, "parameters"):
            self.device = next(obj.parameters()).device
        else:
            self.device = torch.device("cpu")
        self.cuda_types = (torch.Tensor,) + cast_types
        self.stop = threading.Event()
        self.thread = threading.Thread(target=self.run, daemon=True)
        self.thread.start()

    def encode(self, m):
        if self.use_shared_buffer:
            return BufferPickler(self.placeholder._buffer_from_main).dumps(m)
        if self.pickle_messages:
            return pickle.dumps(m)
        return m

    def decode(self, m):
        if self.use_shared_buffer:
            return BufferUnpickler(self.placeholder._buffer_to_main, m, self.device).load()

        if self.pickle_messages:
            return pickle.loads(m)
        return m

    def to_cpu(self, i):
        return i.detach().to(torch.device("cpu")) if isinstance(i, self.cuda_types) else i

    def run(self):
        timeouts = 0
        while not self.stop.is_set() and timeouts < 5 / 1e-5:
            for qi, q in enumerate(self.in_queues):
                try:
                    r = self.decode(q.get(True, 1e-5))
                except queue.Empty:
                    timeouts += 1
                    continue
                except ConnectionError:
                    break
                timeouts = 0
                attr, args, kwargs = r
                if hasattr(self.obj, "lock"):  # TODO: this is not used anywhere?
                    self.obj.lock.acquire()
                f = getattr(self.obj, attr)
                args = [i.to(self.device) if isinstance(i, self.cuda_types) else i for i in args]
                kwargs = {k: i.to(self.device) if isinstance(i, self.cuda_types) else i for k, i in kwargs.items()}
                try:
                    # There's no need to compute gradients, since we can't transfer them back to the worker
                    with torch.no_grad():
                        result = f(*args, **kwargs)
                except Exception as e:
                    result = e
                    exc_str = traceback.format_exc()
                    print(exc_str)
                    try:
                        pickle.dumps(e)
                    except Exception:
                        result = RuntimeError("Exception raised in MPModelProxy, but it cannot be pickled.\n" + exc_str)
                if isinstance(result, (list, tuple)):
                    msg = [self.to_cpu(i) for i in result]
                elif isinstance(result, dict):
                    msg = {k: self.to_cpu(i) for k, i in result.items()}
                else:
                    msg = self.to_cpu(result)
                self.out_queues[qi].put(self.encode(msg))
                if hasattr(self.obj, "lock"):
                    self.obj.lock.release()

    def terminate(self):
        self.stop.set()


def mp_object_wrapper(obj, num_workers, cast_types, pickle_messages: bool = False, sb_size=None):
    """Construct a multiprocessing object proxy for torch DataLoaders so
    that it does not need to be copied in every worker's memory. For example,
    this can be used to wrap a model such that only the main process makes
    cuda calls by forwarding data through the model, or a replay buffer
    such that the new data is pushed in from the worker processes but only the
    main process has to hold the full buffer in memory.

    Parameters
    ----------
    obj: any python object to be proxied (typically a torch.nn.Module or ReplayBuffer)
        Lives in the main process to which method calls are passed
    num_workers: int
        Number of DataLoader workers
    cast_types: tuple
        Types that will be cast to cuda when received as arguments of method calls.
        torch.Tensor is cast by default.
    pickle_messages: bool
        If True, pickle messages sent between processes. This reduces load on shared
        memory, but increases load on CPU. It is recommended to activate this flag if
        encountering "Too many open files"-type errors.
    sb_size: Optional[int]
        If not None, creates a shared buffer of this size for sending tensors between processes.
        Note, this will allocate two buffers of this size (one for sending, the other for receiving).

    Returns
    -------
    placeholder: MPObjectPlaceholder
        A placeholder object whose method calls route arguments to the main process

    """
    return MPObjectProxy(obj, num_workers, cast_types, pickle_messages, sb_size=sb_size)
