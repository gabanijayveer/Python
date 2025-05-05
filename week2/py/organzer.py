import os
import shutil

def organize_files(directory):
    
    files = os.listdir(directory)
    
    
    for file in files:
        file_path = os.path.join(directory, file)
        
   
        if os.path.isfile(file_path):
            
            file_extension = file.split('.')[-1]
            
            if file_extension == file:
                continue
            
            folder_name = os.path.join(directory, file_extension)
            if not os.path.exists(folder_name):
                os.makedirs(folder_name)
            
            new_file_path = os.path.join(folder_name, file)
            
            shutil.move(file_path, new_file_path)
            print(f"Moved: {file} -> {folder_name}")

directory = os.getcwd()  
organize_files(directory)
