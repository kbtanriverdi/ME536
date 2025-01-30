import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk
image_path=None
scale=0.95
size=600

def clear_image():
    # Clear the displayed image
    image_label.config(image="")
    image_label.image = None

#TODO somehow send opened image to classifier
def import_image():
    global image_path
    # Open file dialog to select an image file
    file_path = filedialog.askopenfilename(filetypes=[("Image files", "*.png;*.jpg;*.jpeg;*.bmp;*.gif")])
    if file_path:
        # Open and display the selected image
        img = Image.open(file_path)
        image_path=file_path
        print("image_loaded")
        img = img.resize((int(size*scale), int(size*scale)), Image.LANCZOS)  # Resize to fit the window using LANCZOS
        img_tk = ImageTk.PhotoImage(img)
        image_label.config(image=img_tk)
        image_label.image = img_tk
        with open("temppath.txt", "w") as f:
            f.write(file_path)

def get_image():
    global image_path
    return Image.open(image_path)

if __name__ == "__main__":
    # Create the main window
    root = tk.Tk()
    root.title("Image Selector")
    geometry=str(size)+"x"+str(size)
    #root.geometry("600x600")
    root.geometry(geometry)
    # Create the buttons
    button_frame = tk.Frame(root,padx=15,pady=10)
    button_frame.pack(side=tk.TOP, fill=tk.X, pady=10)

    clear_button = tk.Button(button_frame, text="Clear",font=("Arial", 12), command=clear_image,padx=15,pady=10)
    clear_button.pack(side=tk.LEFT, padx=10)

    import_button = tk.Button(button_frame, text="Import",font=("Arial", 12), command=import_image,padx=15,pady=10)
    import_button.pack(side=tk.LEFT, padx=10)

# Create the label to display the image
    image_label = tk.Label(root)
    image_label.pack(expand=True, fill=tk.BOTH)

# Start the application
    root.mainloop()
