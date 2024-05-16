# Image Enhancement UI
# displays UI to select input and output folders and starts processing image with user friendly UI
#
# requires main.py for funcionality
#
# standalone app (using pyinstaller) is in same folder. Compiled app was not tested on any system other than win32.
#
# created by seven / Jean-Luc Peloquin

import os
import sys
import subprocess
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
import threading
import main  # Importing main.py which includes all functions for processing
import time
from plyer import notification

# opens the folder
def open_folder(path):
    if sys.platform == 'win32': # windows
        os.startfile(path)
    elif sys.platform == 'darwin':  # macOS
        subprocess.run(['open', path])
    else:  # Linux and other Unix-like systems
        subprocess.run(['xdg-open', path])

# creates UI
def create_ui():
    root = tk.Tk()
    root.title("AutoEnhance")

    # app window size
    window_width = 600
    window_height = 200
    screen_width = root.winfo_screenwidth()
    screen_height = root.winfo_screenheight()
    center_x = int(screen_width / 2 - window_width / 2)
    center_y = int(screen_height / 2 - window_height / 2)
    root.geometry(f'{window_width}x{window_height}+{center_x}+{center_y}')

    # app color
    bg_color = "#080808"
    text_color = "#fffdd0"
    button_color = "#080808"
    font_style = ('Helvetica', '12')
    root.configure(background=bg_color)

    # app functionalities
    style = ttk.Style()
    style.theme_use('default')
    style.configure('TProgressbar', background=text_color, thickness=20)
    style.configure('TButton', background=button_color, foreground=text_color, font=font_style)
    style.map('TButton', background=[('active', button_color)], foreground=[('active', text_color)])

    input_dir = tk.StringVar(root)
    output_dir = tk.StringVar(root)
    progress_bar = ttk.Progressbar(root, style='TProgressbar', orient="horizontal", length=100, mode="indeterminate")

    # getting the directory that has the input files
    def select_input_directory():
        directory = filedialog.askdirectory()
        if directory:
            input_dir.set(directory)
            output_directory = os.path.join(os.path.dirname(directory), "output")
            if not os.path.exists(output_directory):
                os.makedirs(output_directory)
            output_dir.set(output_directory)

    # getting the directory that has the outputa files
    def select_output_directory():
        directory = filedialog.askdirectory()
        if directory:
            output_dir.set(directory)

    # starts processing and error checking input/output directories
    def start_processing():
        input_directory = input_dir.get()
        output_directory = output_dir.get()
        if not input_directory:
            messagebox.showerror("Error", "Please select an input directory.")
            return
        progress_bar.pack(fill='x', after=process_button)
        process_button.pack_forget()
        progress_bar.start()
        start_time = time.time()  # Start the timer
        try:
            main.batch_process_images(input_directory, output_directory)
            elapsed_time = time.time() - start_time  # Calculate the elapsed time
            messagebox.showinfo("Completed", "Image processing completed.")
            send_notification(output_directory, elapsed_time)  # Pass the elapsed time to the notification
        finally:
            progress_bar.stop()
            progress_bar.pack_forget()
            process_button.pack(fill='x')
            open_folder(output_directory)

    def send_notification(output_path, elapsed_time):
        formatted_time = f"{elapsed_time:.2f} seconds"  # Format the elapsed time to two decimal places
        notification.notify(
        title="Processing Complete",
        message=f"All images have been processed. Time taken: {formatted_time}",
        app_name="AutoEnhance"
    )

    # making button UI
    tk.Button(root, text="Select Input Folder", command=select_input_directory, height=2, font=font_style, bg=button_color, fg=text_color).pack(fill='x')
    tk.Entry(root, textvariable=input_dir, state='readonly', bg=bg_color, fg=text_color, readonlybackground=bg_color, font=font_style).pack(fill='x')
    tk.Button(root, text="Select Output Folder", command=select_output_directory, height=2, font=font_style, bg=button_color, fg=text_color).pack(fill='x')
    tk.Entry(root, textvariable=output_dir, state='readonly', bg=bg_color, fg=text_color, readonlybackground=bg_color, font=font_style).pack(fill='x')
    process_button = tk.Button(root, text="Start Processing", command=lambda: threading.Thread(target=start_processing).start(), height=2, font=font_style, bg=button_color, fg=text_color)
    process_button.pack(fill='x')
    root.mainloop()

if __name__ == "__main__":
    create_ui()
