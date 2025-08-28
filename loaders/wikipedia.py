from datasets import Dataset
from .base_loader import BaseLoader
from .utils import clean_example


class WIKIPEDIA(BaseLoader):
    """Loader for the Wikipedia dataset."""

    def postprocess(self, dataset): 
        """Format the raw dataset to a common schema.
        
        Parameters
        ----------
        dataset : Dataset
            The input dataset to postprocess.
        subset : str, optional
            The subset of the dataset to process (e.g., "train", "validation", "test").
        split : str
            The split of the dataset to process (e.g., "train", "validation", "test"). Defaults to "train".
        
        Returns
        -------
        Dataset
            The postprocessed dataset with "text", "source", "subset" and "source_split" columns.
        """
        res = {
            "text": list(dataset['text']), 
            "source": [self.source] * len(dataset), 
            "subset": list(dataset['subset']), 
            "source_split": [self.split] * len(dataset)
        }
        return Dataset.from_dict(res)

    #@override
    def load(self):
        """
        Load the dataset from the specified path and apply postprocessing.
        
        Returns
        -------
        Dataset
            The loaded and postprocessed dataset.   
        """
        ds = Dataset.from_parquet(f"{self.path}/wikipedia.parquet")
        tmp_ds = self.postprocess(dataset=ds)
        return tmp_ds.map(clean_example, fn_kwargs={"lower": False, "rm_new_lines": False})
