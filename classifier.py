import tkinter as tk
from backend import *
from PIL import Image
import statcalc

def set_seed(seed):
    """Set random seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

set_seed(42)  # Set seed for reproducibility

class TextClassifierApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Classifier")

        button_pad = {"padx": 15, "pady": 10}

        self.text_label = tk.Label(self.root, text="", font=("Arial", 18))# until here
        self.text_label.pack(pady=15)

        self.button_frame = tk.Frame(self.root)
        self.button_frame.pack(pady=10)

        self.classify_button = tk.Button(self.button_frame, text="Classify",font=("Arial", 10), command=self.classify_text, **button_pad)
        self.classify_button.pack(side=tk.LEFT, padx=10)

        self.correct_button = tk.Button(self.button_frame, text="Clear",font=("Arial", 10), command=self.clear_text,**button_pad)
        self.correct_button.pack(side=tk.LEFT, padx=10)

        self.fix_button = tk.Button(self.button_frame, text="Incorrect/Fix",font=("Arial", 10), command=self.fix_text,**button_pad)
        self.fix_button.pack(side=tk.LEFT, padx=10)

        self.input_box = tk.Entry(self.root, font=("Arial", 14))
        self.input_box.pack(pady=15)

    #TODO somehow take the image from image-open and extract augmented feature vectors and return a classifier result
    def classify_text(self):
        cluster_mapping = {}
        # Open the file and read it line by line
        with open("mapping.txt", "r") as file:
            for line in file:
                # Split the line into key and value
                parts = line.strip().split()
                if len(parts) == 2:
                    key, value = parts
                    # Add the key-value pair to the dictionary
                    cluster_mapping[int(key)] = value

        with open("temppath.txt", "r") as f:
            loaded_image = Image.open(f.read().strip())
        def set_seed(seed):
            """Set random seed for reproducibility."""
            random.seed(seed)
            np.random.seed(seed)
            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(seed)

        set_seed(42)  # Set seed for reproducibility
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        feature_vectors = extract_features([loaded_image]).T
        print(feature_vectors)
        
        # image_path='classifiervectors.csv'
        # if not os.path.exists(image_path):
        #     numpy_array = feature_vectors.T.cpu().numpy()
        #     df = pd.DataFrame(numpy_array)
        #     df.to_csv(image_path, index=False, header=False)
        #     print(f"Tensor values have been exported to '{image_path}'")
        image_path='classifiervectors.csv'
        df = pd.read_csv(image_path, header=None)  # Use `header=None` since `header=False` was used during export
        # Convert the DataFrame to a NumPy array
        clusters = df.to_numpy()
        print(clusters)
        distances=statcalc.mahalanobis_distanceTen(feature_vectors,clusters)
        print(distances)
        min_index=np.argmin(distances)
        print(min_index)
        nearest=distances[min_index]
        print(cluster_mapping)
        print(feature_vectors.shape)
        closenes, valcrit = statcalc.is_close(nearest,dim=6,confidence=float(0.985))
        print("critical value:",valcrit)
        new = pd.DataFrame(statcalc.calculate_statistics(feature_vectors.T.cpu()))
        new.to_csv("temp_image.csv",index=False,header=False)
        if closenes:
            self.text_label.config(text=cluster_mapping[min_index])
        else:
            self.text_label.config(text="Unknown")

    def clear_text(self):
        self.text_label.config(text="")

    #TODO extract augmented feature vectors and calculate their statistical parameters and put into classes list for future reference
    def fix_text(self):
        new_text = self.input_box.get()
        map_path='mapping.txt'
        exist=0
        with open(map_path, "r") as f: 
            for line in f:
                if line.split()[1]==new_text:
                    exist=1
                else:
                    continue
        if new_text:
            df2=pd.read_csv("temp_image.csv",header=None)
            df = pd.read_csv("classifiervectors.csv", header=None)  # Read the file into a DataFrame
            # Add the NumPy array as new columns to the DataFrame
            df = pd.concat([df, df2], axis=1)  # Concatenate along columns

            # Save the updated DataFrame back to the CSV file
            df.to_csv("classifiervectors.csv", index=False, header=False)
            map_path='mapping.txt'
            with open(map_path, "a") as f: 
                f.write(f"{df.shape[1]-1} {new_text}\n")
            self.text_label.config(text=f"Classified as {new_text}")


if __name__ == "__main__":
    size=400
    root = tk.Tk()
    geometry=str(size)+"x"+str(size)
    root.geometry(geometry)
    app = TextClassifierApp(root)
    root.mainloop()
