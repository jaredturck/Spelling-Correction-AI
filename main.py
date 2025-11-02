import customtkinter as ctk
from ai_model import SpellingModel
import torch

class SpellingGUI:
    def __init__(self):
        self.model = SpellingModel().to('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.load_weights()
        self.model.dataset.generate_lookup()
        self.correct_word_cache = {}

    def mirror_text(self, event):
        content = self.input_box.get("1.0", "end-1c")

        print([content])
        words = content.split(' ')
        corrected_words = []
        is_styled = []
        for word in words:
            if word:
                if word in self.correct_word_cache:
                    corrected_words.append(self.correct_word_cache[word])
                    is_styled.append(True)

                elif word in self.model.dataset.word2int_lookup:
                    corrected_words.append(word)
                    is_styled.append(False)

                else:
                    pred = self.model.predict(word)
                    self.correct_word_cache[word] = pred[0]
                    corrected_words.append(pred[0])
                    is_styled.append(True)

        self.output_box.configure(state="normal")
        self.output_box.delete("1.0", "end")

        for i, (word, styled) in enumerate(zip(corrected_words, is_styled)):
            if i > 0:
                self.output_box.insert('end', ' ')

            start = self.output_box.index("end-1c")
            self.output_box.insert('end', word)
            end = self.output_box.index("end-1c")
            if styled:
                self.output_box.tag_add("corrected", start, end)
        
        self.output_box.configure(state="disabled")

    def main_view(self):
        ctk.set_appearance_mode("System")
        ctk.set_default_color_theme("blue")

        app = ctk.CTk()
        app.title("Spelling Correction")
        app.geometry("520x420")

        title = ctk.CTkLabel(app, text="Spelling Correction", font=ctk.CTkFont(size=20, weight="bold"))
        title.pack(pady=(16, 8))

        self.output_box = ctk.CTkTextbox(app, width=480, height=120, wrap="word")
        self.output_box.pack(padx=16, pady=(0, 12), fill="both", expand=False)
        self.output_box.insert("1.0", "")
        self.output_box.configure(state="disabled")
        self.output_box.tag_config("corrected", underline=1, foreground="red")

        self.input_box = ctk.CTkTextbox(app, width=480, height=160, wrap="word")
        self.input_box.pack(padx=16, pady=(0, 16), fill="both", expand=True)

        self.input_box.bind("<KeyRelease>", self.mirror_text)
        self.input_box.bind("<<Paste>>", self.mirror_text)

        app.mainloop()

if __name__ == "__main__":
    gui = SpellingGUI()
    gui.main_view()
