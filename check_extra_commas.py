import tkinter as tk
from tkinter import filedialog

# Hide the root tkinter window
root = tk.Tk()
root.withdraw()

# Prompt to choose your file
file_path = filedialog.askopenfilename(title="Select the CSV file")

# Check for extra commas in each line
with open(file_path, "r", encoding="utf-8") as f:
    for i, line in enumerate(f, start=1):
        # Count commas
        if line.count(",") > 1:
            print(f"Line {i} → {line.strip()}")
