import os
import sqlite3
from typing import Iterable

import torch


class SQLiteLogHook:
    def __init__(self, log_dir, ctx) -> None:
        self.log = None  # Only initialized in __call__, which will occur inside the worker
        self.log_dir = log_dir
        self.ctx = ctx
        self.data_labels = None

    def __call__(self, trajs, rewards, flat_rewards, cond_info):
        if self.log is None:
            worker_info = torch.utils.data.get_worker_info()
            self._wid = worker_info.id if worker_info is not None else 0
            os.makedirs(self.log_dir, exist_ok=True)
            self.log_path = f"{self.log_dir}/generated_mols_{self._wid}.db"
            self.log = SQLiteLog()
            self.log.connect(self.log_path)

        if hasattr(self.ctx, "object_to_log_repr"):
            mols = [self.ctx.object_to_log_repr(t["result"]) if t["is_valid"] else "" for t in trajs]
        else:
            mols = [""] * len(trajs)

        flat_rewards = flat_rewards.reshape((len(flat_rewards), -1)).data.numpy().tolist()
        rewards = rewards.data.numpy().tolist()
        preferences = cond_info.get("preferences", torch.zeros((len(mols), 0))).data.numpy().tolist()
        focus_dir = cond_info.get("focus_dir", torch.zeros((len(mols), 0))).data.numpy().tolist()
        logged_keys = [k for k in sorted(cond_info.keys()) if k not in ["encoding", "preferences", "focus_dir"]]

        data = [
            [mols[i], rewards[i]]
            + flat_rewards[i]
            + preferences[i]
            + focus_dir[i]
            + [cond_info[k][i].item() for k in logged_keys]
            for i in range(len(trajs))
        ]
        if self.data_labels is None:
            self.data_labels = (
                ["smi", "r"]
                + [f"fr_{i}" for i in range(len(flat_rewards[0]))]
                + [f"pref_{i}" for i in range(len(preferences[0]))]
                + [f"focus_{i}" for i in range(len(focus_dir[0]))]
                + [f"ci_{k}" for k in logged_keys]
            )

        self.log.insert_many(data, self.data_labels)
        return {}


class SQLiteLog:
    def __init__(self, timeout=300):
        """Creates a log instance, but does not connect it to any db."""
        self.is_connected = False
        self.db = None
        self.timeout = timeout

    def connect(self, db_path: str):
        """Connects to db_path

        Parameters
        ----------
        db_path: str
            The sqlite3 database path. If it does not exist, it will be created.
        """
        self.db = sqlite3.connect(db_path, timeout=self.timeout)
        cur = self.db.cursor()
        self._has_results_table = len(
            cur.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='results'").fetchall()
        )
        cur.close()

    def _make_results_table(self, types, names):
        type_map = {str: "text", float: "real", int: "real"}
        col_str = ", ".join(f"{name} {type_map[t]}" for t, name in zip(types, names))
        cur = self.db.cursor()
        cur.execute(f"create table results ({col_str})")
        self._has_results_table = True
        cur.close()

    def insert_many(self, rows, column_names):
        assert all(
            [isinstance(x, str) or not isinstance(x, Iterable) for x in rows[0]]
        ), "rows must only contain scalars"
        if not self._has_results_table:
            self._make_results_table([type(i) for i in rows[0]], column_names)
        cur = self.db.cursor()
        cur.executemany(f'insert into results values ({",".join("?"*len(rows[0]))})', rows)  # nosec
        cur.close()
        self.db.commit()
