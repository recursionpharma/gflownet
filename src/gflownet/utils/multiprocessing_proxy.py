import pickle
import queue
import threading
import traceback

import torch
import torch.multiprocessing as mp


class MPObjectPlaceholder:
    """This class can be used for example as a model or dataset placeholder
    in a worker process, and translates calls to the object-placeholder into
    queries for the main process to execute on the real object."""

    def __init__(self, in_queues, out_queues, pickle_messages=False):
        self.qs = in_queues, out_queues
        self.device = torch.device("cpu")
        self.pickle_messages = pickle_messages
        self._is_init = False

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

    def __init__(self, obj, num_workers: int, cast_types: tuple, pickle_messages: bool = False):
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
        """
        self.in_queues = [mp.Queue() for i in range(num_workers + 1)]  # type: ignore
        self.out_queues = [mp.Queue() for i in range(num_workers + 1)]  # type: ignore
        self.pickle_messages = pickle_messages
        self.placeholder = MPObjectPlaceholder(self.in_queues, self.out_queues, pickle_messages)
        self.obj = obj
        if hasattr(obj, "parameters"):
            self.device = next(obj.parameters()).device
        else:
            self.device = torch.device("cpu")
        self.cuda_types = (torch.Tensor,) + cast_types
        self.stop = threading.Event()
        self.thread = threading.Thread(target=self.run, daemon=True)
        self.thread.start()

    def __del__(self):
        self.stop.set()

    def encode(self, m):
        if self.pickle_messages:
            return pickle.dumps(m)
        return m

    def decode(self, m):
        if self.pickle_messages:
            return pickle.loads(m)
        return m

    def to_cpu(self, i):
        return i.detach().to(torch.device("cpu")) if isinstance(i, self.cuda_types) else i

    def run(self):
        while not self.stop.is_set():
            for qi, q in enumerate(self.in_queues):
                try:
                    r = self.decode(q.get(True, 1e-5))
                except queue.Empty:
                    continue
                except ConnectionError:
                    break
                attr, args, kwargs = r
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


def mp_object_wrapper(obj, num_workers, cast_types, pickle_messages: bool = False):
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
    return MPObjectProxy(obj, num_workers, cast_types, pickle_messages).placeholder
