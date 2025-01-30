from backend import *
import statcalc

def train():
    from PIL import Image
    import random
    # Load your images here (as PIL.Image objects)
    mp="D:\\barış karışık yedekler\\okul3\\7.yy\\ME536\\proje\\demo\\data\\"
    text_file = 'D:\\barış karışık yedekler\\okul3\\7.yy\\ME536\\proje\\demo\\codes\\f16.txt'
    image_paths = []
    with open(text_file, 'r') as file:
        for line in file:
        # Clean the line to remove any leading/trailing whitespace
            line = line.split()[0]

        # Check if the line is a valid number (optional, based on your text file structure)
            if line.isdigit():
                image_name = f"{line}.jpg"
                image_path = os.path.join(mp, image_name)
            
            # Check if the image exists
            if os.path.exists(image_path):
                image_paths.append(image_name)
            else:
                print(f"Image {image_name} does not exist.")
    """
    paths = random.sample(image_paths, min(1, len(image_paths)))
    print("paths:",paths)
    images = [Image.open(mp+path) for path in paths]
    """
    images = [Image.open(mp+path) for path in image_paths]
    
    feature_vectors = extract_features(images)
    feature_vectors = statcalc.calculate_statistics(feature_vectors).T
    print("Feature vectors shape:", feature_vectors.shape)
    print(feature_vectors)
    image_path='classifiervectors.csv'
    if not os.path.exists(image_path):
        numpy_array = feature_vectors.T.cpu().numpy()
        df = pd.DataFrame(numpy_array)
        df.to_csv(image_path, index=False, header=False)
        print(f"Tensor values have been exported to '{image_path}'")
    else:
        print(f"'{image_path}' already exists.")
    

    map_path='mapping.txt'
    if not os.path.exists(map_path):
        with open(map_path, "w") as f: 
            f.write("0 F16\n")
        print(f"Tensor values have been exported to '{image_path}'")

if __name__ == "__main__":
    train()
