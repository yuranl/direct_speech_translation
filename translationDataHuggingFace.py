import torch
import os
import json
from torch.utils.data import Dataset, DataLoader

class translationDataset(Dataset):
    """Face Landmarks dataset."""

    def __init__(self, datafolder):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.source = []
        self.target = []
        for filename in os.listdir(datafolder):
            if filename.endswith(".json"):
                with open(datafolder + filename, encoding="utf8") as f:
                    json_data = f.read()
                    json_data = json_data.replace('\n', ',')[:-1]
                    data = json.loads("["+json_data+"]")
                    for item in data:
                        self.source.append(item['transcript'])
                        self.target.append(item['translation'])


    def __len__(self):
        assert(len(self.source) == len(self.target))
        return len(self.source)

    def __getitem__(self, idx):
        # if torch.is_tensor(idx):
        #     idx = idx.tolist()
        #
        # img_name = os.path.join(self.root_dir,
        #                         self.landmarks_frame.iloc[idx, 0])
        # image = io.imread(img_name)
        # landmarks = self.landmarks_frame.iloc[idx, 1:]
        # landmarks = np.array([landmarks])
        # landmarks = landmarks.astype('float').reshape(-1, 2)
        # sample = {'image': image, 'landmarks': landmarks}
        #
        # if self.transform:
        #     sample = self.transform(sample)
        #
        return {'source': self.source[idx], 'targer':self.target[idx]}





testdataset = translationDataset('./transcription_translation/')