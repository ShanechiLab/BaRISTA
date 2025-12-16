import json
import os
from typing import Tuple

class FileProgressTracker:
    """Manage loading and storing latest completely processed file index

    This class save information required to continue processing in a file.
    The file structure will be:
    {
        [experiment]: {
            [self._file_ind_key]: int,
            [self._ending_ind_key]: int,
            [self._segment_id_key]: int
        }
    }
    """

    def __init__(self, save_path: str, experiment: str):
        self.path = save_path
        self.experiment = experiment
        self._file_ind_key = "file_ind"
        self._ending_ind_key = "ending_ind"
        self._segment_id_key = "segment_id"
        self._completed_key = "is_completed"

    def _load_file(self) -> dict:
        """Load processing info from file

        Returns:
            A dictionary having structure as descripted in the class info
        """
        data = {}
        if os.path.exists(self.path):
            with open(self.path) as f:
                data = json.load(f)

        if self.experiment not in data:
            data[self.experiment] = {
                self._file_ind_key: 0,
                self._ending_ind_key: 0,
                self._segment_id_key: -1,
                self._completed_key: False,
            }

        return data

    def _update_file(self, update_dict: dict) -> None:
        """Update specified keys in file"""

        data = self._load_file()
        data[self.experiment].update(update_dict)

        with open(self.path, "w+") as f:
            json.dump(data, f)

    def get_last_file_ind(self) -> Tuple[int, int, int]:
        """Get last file that was processed for this experiment

        Returns:
            A tuple containing file index, ending index in the file, and the segment number of the last processed file
        """
        data = self._load_file()
        return (
            data[self.experiment][self._file_ind_key],
            data[self.experiment][self._ending_ind_key],
            data[self.experiment][self._segment_id_key],
        )

    def update_last_file_ind(
        self, file_ind: int, ending_ind: int, segment_id: int
    ) -> None:
        """Update last file processed info in this experiment without changing other info in file if necessary"""

        self._update_file(
            {
                self._file_ind_key: file_ind,
                self._ending_ind_key: ending_ind,
                self._segment_id_key: segment_id,
            }
        )

    def mark_completion_status(self, completed: bool = True) -> None:
        self._update_file({self._completed_key: completed})

    def is_completed(self) -> bool:
        data = self._load_file()
        return data[self.experiment].get(self._completed_key, False)

    def reset_process(self) -> None:
        """Reset file processing status"""
        self.mark_completion_status(completed=False)
        self.update_last_file_ind(0, 0, -1)
