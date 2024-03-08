import pickle
import queue
import threading
import traceback
from itertools import chain

import numpy as np
import torch
import torch.multiprocessing as mp
from torch_geometric.data import Batch
import warnings

from gflownet.envs.graph_building_env import GraphActionCategorical


class SharedPinnedBuffer:
    def __init__(self, size):
        self.size = size
        self.buffer = torch.empty(size, dtype=torch.uint8)
        self.buffer.share_memory_()
        self.lock = mp.Lock()
        self.do_unreg = False

        if not self.buffer.is_pinned():
            # Sometimes torch will create an already pinned (page aligned) buffer, so we don't need to 
            # pin it again; doing so will raise a CUDA error
            cudart = torch.cuda.cudart()
            r = cudart.cudaHostRegister(self.buffer.data_ptr(), self.buffer.numel() * self.buffer.element_size(), 0)
            assert r == 0
            self.do_unreg = True  # But then we need to unregister it later
        assert self.buffer.is_shared()
        assert self.buffer.is_pinned()

    def __del__(self):
        if self.do_unreg and torch.utils.data.get_worker_info() is None:
            cudart = torch.cuda.cudart()
            r = cudart.cudaHostUnregister(self.buffer.data_ptr())
            assert r == 0


class BatchDescriptor:
    def __init__(self, names, types, shapes, size, other):
        self.names = names
        self.types = types
        self.shapes = shapes
        self.size = size
        self.other = other


class ResultDescriptor:
    def __init__(self, names, types, shapes, size, gac_attrs):
        self.names = names
        self.types = types
        self.shapes = shapes
        self.size = size
        self.gac_attrs = gac_attrs


def prod(l):
    p = 1
    for i in l:
        p *= i
    return p


def put_into_batch_buffer(batch, buffer):
    names = []
    types = []
    shapes = []
    offset = 0
    others = {}
    for k, v in chain(batch._store.items(), (("_slice_dict_" + k, v) for k, v in batch._slice_dict.items())):
        if not isinstance(v, torch.Tensor):
            try:
                v = torch.as_tensor(v)
            except Exception as e:
                others[k] = v
                continue
        names.append(k)
        types.append(v.dtype)
        shapes.append(tuple(v.shape))
        numel = v.numel() * v.element_size()
        buffer[offset : offset + numel] = v.view(-1).view(torch.uint8)
        offset += numel
        offset += (8 - offset % 8) % 8  # align to 8 bytes
        if offset > buffer.shape[0]:
            raise ValueError(
                f"Offset {offset} exceeds buffer size {buffer.shape[0]}. Try increasing `cfg.mp_buffer_size`."
            )
    return BatchDescriptor(names, types, shapes, offset, others)


def resolve_batch_buffer(descriptor, buffer, device):
    offset = 0
    batch = Batch()
    batch._slice_dict = {}
    # Seems legit to send just a 0-starting slice, because it should be pinned as well (and timing this vs sending
    # the whole buffer, it seems to be the marginally faster option)
    cuda_buffer = buffer[: descriptor.size].to(device)

    for name, dtype, shape in zip(descriptor.names, descriptor.types, descriptor.shapes):
        numel = prod(shape) * dtype.itemsize
        if name.startswith("_slice_dict_"):
            batch._slice_dict[name[12:]] = cuda_buffer[offset : offset + numel].view(dtype).view(shape)
        else:
            setattr(batch, name, cuda_buffer[offset : offset + numel].view(dtype).view(shape))
        offset += numel
        offset += (8 - offset % 8) % 8  # align to 8 bytes

    for k, v in descriptor.other.items():
        setattr(batch, k, v)
    return batch


def put_into_result_buffer(result, buffer):
    gac_names = ["logits", "batch", "slice", "masks"]
    gac, tensor = result
    buffer[: tensor.numel() * tensor.element_size()] = tensor.view(-1).view(torch.uint8)
    offset = tensor.numel() * tensor.element_size()
    offset += (8 - offset % 8) % 8  # align to 8 bytes
    names = ["@per_graph_out"]
    types = [tensor.dtype]
    shapes = [tensor.shape]
    for name in gac_names:
        tensors = getattr(gac, name)
        for i, x in enumerate(tensors):
            numel = x.numel() * x.element_size()
            if numel > 0:
                # We need this for a funny reason
                # torch.zeros(0)[::2] has a stride of (2,), and is contiguous according to torch
                # so, flattening it and then reshaping it will not change the stride, which will
                # make view(uint8) complain that the strides are not compatible.
                # The batch[::2] happens when creating the categorical and deduplicate_edge_index is True
                buffer[offset : offset + numel] = x.flatten().view(torch.uint8)
                offset += numel
                offset += (8 - offset % 8) % 8  # align to 8 bytes
                if offset > buffer.shape[0]:
                    raise ValueError(f"Offset {offset} exceeds buffer size {buffer.shape[0]}")
            names.append(f"{name}@{i}")
            types.append(x.dtype)
            shapes.append(tuple(x.shape))
    return ResultDescriptor(names, types, shapes, offset, (gac.num_graphs, gac.keys, gac.types))


def resolve_result_buffer(descriptor, buffer, device):
    # TODO: models can return multiple GraphActionCategoricals, but we only support one for now
    # Would be nice to have something generic (and recursive?)
    offset = 0
    tensor = buffer[: descriptor.size].to(device)
    if tensor.device == device:  # CPU to CPU
        # I think we need this? Otherwise when we release the lock, the memory might be overwritten
        tensor = tensor.clone()
    # Maybe make this a static method, or just overload __new__?
    gac = GraphActionCategorical.__new__(GraphActionCategorical)
    gac.num_graphs, gac.keys, gac.types = descriptor.gac_attrs
    gac.dev = device
    gac.logprobs = None
    gac._epsilon = 1e-38

    gac_names = ["logits", "batch", "slice", "masks"]
    for i in gac_names:
        setattr(gac, i, [None] * len(gac.types))

    for name, dtype, shape in zip(descriptor.names, descriptor.types, descriptor.shapes):
        numel = prod(shape) * dtype.itemsize
        if name == "@per_graph_out":
            per_graph_out = tensor[offset : offset + numel].view(dtype).view(shape)
        else:
            name, index = name.split("@")
            index = int(index)
            if name in gac_names:
                getattr(gac, name)[index] = tensor[offset : offset + numel].view(dtype).view(shape)
            else:
                raise ValueError(f"Unknown result descriptor name: {name}")
        offset += numel
        offset += (8 - offset % 8) % 8  # align to 8 bytes
    return gac, per_graph_out


class MPObjectPlaceholder:
    """This class can be used for example as a model or dataset placeholder
    in a worker process, and translates calls to the object-placeholder into
    queries for the main process to execute on the real object."""

    def __init__(self, in_queues, out_queues, pickle_messages=False, batch_buffer_size=None):
        self.qs = in_queues, out_queues
        self.device = torch.device("cpu")
        self.pickle_messages = pickle_messages
        self._is_init = False
        self.batch_buffer_size = batch_buffer_size
        if batch_buffer_size is not None:
            self._batch_buffer = SharedPinnedBuffer(batch_buffer_size)
            self._result_buffer = SharedPinnedBuffer(batch_buffer_size)

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
        if self.pickle_messages:
            return pickle.dumps(m)
        return m

    def decode(self, m):
        if self.pickle_messages:
            m = pickle.loads(m)
        if isinstance(m, Exception):
            print("Received exception from main process, reraising.")
            raise m
        if isinstance(m, ResultDescriptor):
            m = resolve_result_buffer(m, self._result_buffer.buffer, self.device)
            self._result_buffer.lock.release()
        return m

    def __getattr__(self, name):
        def method_wrapper(*a, **kw):
            self._check_init()
            self.in_queue.put(self.encode((name, a, kw)))
            return self.decode(self.out_queue.get())

        return method_wrapper

    def __call__(self, *a, **kw):
        self._check_init()
        if self.batch_buffer_size and len(a) and isinstance(a[0], Batch):
            # The lock will be released by the consumer of this buffer once the memory has been transferred to CUDA
            self._batch_buffer.lock.acquire()
            batch_descriptor = put_into_batch_buffer(a[0], self._batch_buffer.buffer)
            a = (batch_descriptor,) + a[1:]
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

    def __init__(self, obj, num_workers: int, cast_types: tuple, pickle_messages: bool = False, bb_size=None):
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
        bb_size: Optional[int]
            batch buffer size
        """
        self.in_queues = [mp.Queue() for i in range(num_workers + 1)]  # type: ignore
        self.out_queues = [mp.Queue() for i in range(num_workers + 1)]  # type: ignore
        self.pickle_messages = pickle_messages
        self.placeholder = MPObjectPlaceholder(self.in_queues, self.out_queues, pickle_messages, bb_size)
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
        if self.pickle_messages:
            return pickle.dumps(m)
        if (
            self.placeholder.batch_buffer_size
            and isinstance(m, (list, tuple))
            and len(m) == 2
            and isinstance(m[0], GraphActionCategorical)
            and isinstance(m[1], torch.Tensor)
        ):
            self.placeholder._result_buffer.lock.acquire()
            return put_into_result_buffer(m, self.placeholder._result_buffer.buffer)
        return m

    def decode(self, m):
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
                if self.placeholder.batch_buffer_size and len(args) and isinstance(args[0], BatchDescriptor):
                    batch = resolve_batch_buffer(args[0], self.placeholder._batch_buffer.buffer, self.device)
                    args = (batch,) + args[1:]
                    # Should this release happen after the call to f()? Are we at risk of overwriting memory that
                    # is still being used by CUDA?
                    self.placeholder._batch_buffer.lock.release()
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

    def terminate(self):
        self.stop.set()


def mp_object_wrapper(obj, num_workers, cast_types, pickle_messages: bool = False, bb_size=None):
    """Construct a multiprocessing object proxy for torch DataLoaders so
    that it does not need to be copied in every worker's memory. For example,
    this can be used to wrap a model such that only the main process makes
    cuda calls by forwarding data through the model, or a replay buffer
    such that the new data is pushed in from the worker processes but only the
    main process has to hold the full buffer in memory.
                    self.out_queues[qi].put(self.encode(msg))
                elif isinstance(result, dict):
                    msg = {k: self.to_cpu(i) for k, i in result.items()}
                    self.out_queues[qi].put(self.encode(msg))
                else:
                    msg = self.to_cpu(result)
                    self.out_queues[qi].put(self.encode(msg))

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

    Returns
    -------
    placeholder: MPObjectPlaceholder
        A placeholder object whose method calls route arguments to the main process

    """
    return MPObjectProxy(obj, num_workers, cast_types, pickle_messages, bb_size=bb_size)
