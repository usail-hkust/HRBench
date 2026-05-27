"""
Unified dataset loader for all 5 benchmark datasets.
Provides a consistent interface regardless of dataset format.
"""
import json
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
from pathlib import Path

from src.config.base_config import DATASET_REGISTRY, DATA_RAW_DIR


@dataclass
class Problem:
    """A single benchmark problem."""
    id: int                              # Index in the dataset
    problem: str                         # Problem text
    answer: Optional[str] = None         # Ground truth answer (math/science)
    solution: Optional[str] = None       # Reference solution text
    test_cases: Optional[List[Dict]] = None  # Test cases (code)
    domain: str = "math"                 # math / science / code
    dataset: str = ""                    # Dataset name
    raw: Dict[str, Any] = field(default_factory=dict)  # Raw data for any extras


class BenchmarkDataset:
    """
    Unified dataset loader.

    Usage:
        ds = BenchmarkDataset("math500")
        for problem in ds:
            print(problem.problem, problem.answer)
    """

    def __init__(self, dataset_id: str, subset: Optional[range] = None):
        """
        Args:
            dataset_id: Key in DATASET_REGISTRY (e.g., "math500", "aime2025")
            subset: Optional range for data sharding (e.g., range(0, 100))
        """
        assert dataset_id in DATASET_REGISTRY, (
            f"Unknown dataset: {dataset_id}. Available: {list(DATASET_REGISTRY.keys())}"
        )
        self.dataset_id = dataset_id
        self.config = DATASET_REGISTRY[dataset_id]
        self._raw_data = self._load()
        if subset is not None:
            self._raw_data = self._raw_data[subset.start : subset.stop]
        self._problems = self._parse()

    def _load(self) -> list:
        path = self.config["path"]
        if not Path(path).exists():
            raise FileNotFoundError(f"Dataset not found: {path}")
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)

    def _parse(self) -> List[Problem]:
        problems = []
        pk = self.config["problem_key"]
        ak = self.config.get("answer_key")
        sk = self.config.get("solution_key")
        tk = self.config.get("test_cases_key")
        domain = self.config["domain"]

        for i, item in enumerate(self._raw_data):
            problems.append(
                Problem(
                    id=i,
                    problem=item[pk],
                    answer=item.get(ak) if ak else None,
                    solution=item.get(sk) if sk else None,
                    test_cases=item.get(tk) if tk else None,
                    domain=domain,
                    dataset=self.dataset_id,
                    raw=item,
                )
            )
        return problems

    def __len__(self) -> int:
        return len(self._problems)

    def __getitem__(self, idx) -> Problem:
        return self._problems[idx]

    def __iter__(self):
        return iter(self._problems)

    def shard(self, gpu_id: int, num_gpus: int) -> "BenchmarkDataset":
        """Create a sharded view for multi-GPU parallel inference."""
        total = len(self._raw_data)
        chunk_size = (total + num_gpus - 1) // num_gpus
        start = gpu_id * chunk_size
        end = min((gpu_id + 1) * chunk_size, total)
        ds = BenchmarkDataset.__new__(BenchmarkDataset)
        ds.dataset_id = self.dataset_id
        ds.config = self.config
        ds._raw_data = self._raw_data[start:end]
        ds._problems = self._parse.__func__(ds)  # re-parse the shard
        return ds

    @property
    def domain(self) -> str:
        return self.config["domain"]

    @property
    def eval_method(self) -> str:
        return self.config["eval_method"]


def load_all_datasets() -> Dict[str, BenchmarkDataset]:
    """Load all registered datasets. Returns dict: dataset_id -> BenchmarkDataset."""
    return {k: BenchmarkDataset(k) for k in DATASET_REGISTRY}


def list_datasets() -> List[str]:
    """List all available dataset IDs."""
    return list(DATASET_REGISTRY.keys())
