class MNISTDataset(Dataset):
    def __init__(self, data_df:pd.DataFrame, transform=None, is_test=False):
        super(MNISTDataset, self).__init__()
        dataset = []
            
        for i, row in tqdm(data_df.iterrows(), total=len(data_df)):
            data = row.to_numpy()
            if is_test:
                label = -1
                image = data.reshape(28, 28).astype(np.uint8)
            else:
                label = data[0]
                image = data[1:].reshape(28, 28).astype(np.uint8)
            
            if transform is not None:
                image = transform(image)
                    
            dataset.append((image, label))
        self.dataset = dataset
        self.transform = transform
        self.is_test = is_test
    
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, i):
        return self.dataset[i]