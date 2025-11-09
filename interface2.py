import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import os
import json

class ImageApp5:
    def __init__(self, root):
        self.root = root
        self.root.title("Классификация морских животных")
        
        self.base_dir = 'models_and_checkpoints'
        self.class_names = [
            'Clams', 'Clown Fish', 'Crabs', 'Dolphin', 'Eel', 'Jellyfish', 'Limacina(морской черт)','Lobster',
            'Magnapinna(длиннорукий кальмар)', 'Minoga', 'Octopus', 'Penguin', 'Physalia physalis (португальский кораблик)',
            'Sea Rays', 'Sea Turtle', 'Sea Urchins','Sea angel', 'Seahorse', 'Seal',
            'Sharks', 'Shrimp', 'Squid', 'Starfish', 'Whale'
        ]
        
        self.model = None
        self.image_path = None
        self.model_info = f"Подробная информация:\n\n"  
        self.checkpoint_info =f""
    
        self.create_widgets()
    
    def create_widgets(self):
     
        left_frame = tk.Frame(self.root)
        left_frame.grid(row=0, column=0, sticky="nw", padx=10)

        self.upload_btn = tk.Button(left_frame, text="Загрузить изображение", command=self.upload_image, width=30)
        self.upload_btn.grid(row=0, column=0, pady=10,sticky="w")

        self.image_label = tk.Label(left_frame)
        self.image_label.grid(row=1, column=0, pady=10,sticky="w")
        self.image_label.grid_forget()

        self.model_var = tk.StringVar(self.root)
        self.model_options = self.get_model_directories()
        self.model_menu = tk.OptionMenu(left_frame, self.model_var, *self.model_options)
        self.model_var.set("Выберите модель")
        self.model_menu.config(width=25)
        self.model_menu.grid(row=2, column=0, pady=20,sticky="w")
        self.model_menu.grid_forget()

        self.load_model_btn = tk.Button(left_frame, text="Загрузить модель", command=self.load_selected_model, width=30)
        self.load_model_btn.grid(row=3, column=0, pady=20,sticky="w")
        self.load_model_btn.grid_forget()

        self.checkpoint_var = tk.StringVar(self.root)
        self.checkpoint_menu = tk.OptionMenu(left_frame, self.checkpoint_var, "Выберите чекпоинт")
        self.checkpoint_menu.config(width=25)
        self.checkpoint_var.set("Выберите чекпоинт")
        self.checkpoint_menu.grid(row=4, column=0, pady=20,sticky="w")
        self.checkpoint_menu.grid_forget()

 
        self.load_checkpoint_btn = tk.Button(left_frame, text="Загрузить чекпоинт", command=self.load_selected_checkpoint, width=30)
        self.load_checkpoint_btn.grid(row=5, column=0, pady=20,sticky="w")
        self.load_checkpoint_btn.grid_forget()

  
        self.classify_btn = tk.Button(left_frame, text="Классифицировать изображение", command=self.classify_image, width=30)
        self.classify_btn.grid(row=6, column=0, pady=20,sticky="w")
        self.classify_btn.grid_forget()

        self.show_info_btn = tk.Button(left_frame, text="Показать подробную информацию", command=self.show_model_info, width=30)
        self.show_info_btn.grid(row=7, column=0, pady=20,sticky="w")
        self.show_info_btn.grid_forget()

        self.result_label = tk.Label(left_frame, text="Результат классификации:", width=30)
        self.result_label.grid(row=8, column=0, pady=20,sticky="w")
        self.result_label.grid_forget()

        self.right_frame = tk.Frame(self.root,width=50)
        self.right_frame.grid(row=0, column=1, padx=30,pady=10, sticky="nw")
        
        self.model_info_label = tk.Label(self.right_frame, text="Информация о модели:", width=60)
        self.model_info_label.grid(row=0, column=1, padx=30,pady=10,sticky="w")
        self.model_info_label.grid_forget()
        self.checkpoint_info_label = tk.Label(self.right_frame, text="Информация о чекпоинте:", width=60)
        self.checkpoint_info_label.grid(row=1, column=1, padx=30,pady=10,sticky="w")
        self.checkpoint_info_label.grid_forget()
        self.right_frame.grid_forget()

    def upload_image(self):
 
        self.right_frame.grid_forget()
        self.model_info_label.grid_forget()
        self.checkpoint_info_label.grid_forget()
        self.result_label.grid_forget()
        self.image_path = filedialog.askopenfilename(filetypes=[("Image files", "*.jpg *.jpeg *.png")])
        if self.image_path:
            img = Image.open(self.image_path)
            img.thumbnail((224, 224), Image.LANCZOS)
            self.img_tk = ImageTk.PhotoImage(img)
            self.image_label.config(image=self.img_tk)
            self.image_label.grid()
            self.show_model_options()

    def get_model_directories(self):
 
        models_path = os.path.join(self.base_dir, 'models')
        return [d for d in os.listdir(models_path) if os.path.isdir(os.path.join(models_path, d))]

    def show_model_options(self):
        self.model_menu.grid()
        self.load_model_btn.grid()
        self.checkpoint_menu.grid()
        self.load_checkpoint_btn.grid()
        self.classify_btn.grid()
        self.show_info_btn.grid()

    def load_selected_model(self):
        selected_model_dir = self.model_var.get()
        if selected_model_dir != "Выберите модель":
            model_path = os.path.join(self.base_dir, 'models', selected_model_dir)
            self.clear_model_info() 
            self.clear_checkpoint_info() 
            model_file = next((f for f in os.listdir(model_path) if f.endswith('.keras') or f.endswith('.h5')), None)
            if model_file:
                self.model = load_model(os.path.join(model_path, model_file))
                messagebox.showinfo("Загрузка модели", f"Модель '{model_file}' успешно загружена.")
                
                # Добавление сжатой информации о модели
                self.model_info += f"Модель: {model_file}\n"
                self.model_info += f"Количество слоев: {len(self.model.layers)}\n"
                self.model_info += f"Количество параметров: {self.model.count_params()}\n"
                
                # # Если модель компилирована, вывести информацию о компиляции
                if hasattr(self.model, 'optimizer') and hasattr(self.model, 'loss') and hasattr(self.model, 'metrics'):
                     self.model_info += f"Функция потерь: {self.model.loss}\n"
               
                    
                
                self.load_checkpoint_options(model_path)
            else:
                messagebox.showwarning("Ошибка", "Файл модели не найден.")

    def load_checkpoint_options(self, model_path):
        checkpoint_dir = os.path.join(model_path, 'checkpoints')
        checkpoints = [
            f for f in os.listdir(checkpoint_dir) if f.endswith('.keras') or f.endswith('.h5')
        ] if os.path.isdir(checkpoint_dir) else []
        self.checkpoint_menu['menu'].delete(0, 'end')
        for checkpoint in checkpoints:
            self.checkpoint_menu['menu'].add_command(label=checkpoint, command=tk._setit(self.checkpoint_var, checkpoint))
        self.checkpoint_var.set("Выберите чекпоинт")

    def load_selected_checkpoint(self):
        selected_checkpoint = self.checkpoint_var.get()
        selected_model_dir = self.model_var.get()
        if self.model and selected_checkpoint != "Выберите чекпоинт":
            checkpoint_path = os.path.join(self.base_dir, 'models', selected_model_dir, 'checkpoints', selected_checkpoint)
            self.model.load_weights(checkpoint_path)
            messagebox.showinfo("Загрузка чекпоинта", f"Чекпоинт '{selected_checkpoint}' успешно загружен.")
            self.clear_checkpoint_info() 
            json_path = checkpoint_path.replace('.keras', '.json').replace('.h5', '.json')
            if os.path.exists(json_path):
                with open(json_path, 'r') as json_file:
                    checkpoint_data = json.load(json_file)
                    self.checkpoint_info += (
                        f"Информация о чекпоинте:\n\n"
                        f"Загружен чекпоинт: {selected_checkpoint}\n"
                        f"Эпоха: {checkpoint_data['epoch']}\n"
                        f"Потери на валидации: {checkpoint_data['val_loss']:.4f}\n"
                        f"Точность на валидации: {checkpoint_data['val_accuracy']:.4f}\n"
                        f"Потери на обучении: {checkpoint_data['train_loss']:.4f}\n"
                        f"Точность на обучении: {checkpoint_data['train_accuracy']:.4f}\n"
                    )
            else:
                self.checkpoint_info+= f"\nИнформация о чекпоинте отсутствует.\n"
            

    def classify_image(self):

        if self.model and self.image_path:
            img = image.load_img(self.image_path, target_size=(224, 224))
            img_array = image.img_to_array(img) / 255.0
            img_array = np.expand_dims(img_array, axis=0)
            predictions = self.model.predict(img_array)
            pred_class = np.argmax(predictions)
            confidence = np.max(predictions)
            self.result_label.config(text=f"Класс: {self.class_names[pred_class]}\nТочность предсказания: {confidence:.2f}")
            self.result_label.grid()
        else:
            messagebox.showwarning("Ошибка", "Пожалуйста, загрузите модель и изображение перед классификацией.")
    
    def show_model_info(self):
        self.right_frame.grid(row=0, column=1, padx=30, pady=10, sticky="nw")
        self.model_info_label.config(text=self.model_info,anchor="w", justify="left",pady=10,padx=5)
        self.checkpoint_info_label.config(text=self.checkpoint_info,anchor="w", justify="left",pady=10,padx=5)
        self.model_info_label.grid(row=0, column=1, padx=30,sticky="w")
        self.checkpoint_info_label.grid(row=1, column=1, padx=30,sticky="w")
        

    def update_model_info(self):
        self.model_info_label.config(text=self.model_info,anchor="w", justify="left",pady=10,padx=5)
        self.checkpoint_info_label.config(text=self.checkpoint_info,anchor="w", justify="left",pady=10,padx=5)
    def clear_model_info(self):
        self.model_info = "Информация о модели:\n\n"
    def clear_checkpoint_info(self):
        self.checkpoint_info = ""   
root = tk.Tk()
app = ImageApp5(root)
root.mainloop()