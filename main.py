import difflib
import customtkinter as ctk
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

class SpellingGUI:
    def __init__(self):
        self.model_path = './qwen_spelling_model/final'
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.after_id = None

        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_path,
        )

        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_path,
            dtype=torch.bfloat16 if self.device == 'cuda' else torch.float32,
        ).to(self.device)

        self.model.eval()

        self.pipe = pipeline(
            'text-generation',
            model=self.model,
            tokenizer=self.tokenizer,
        )

    def mirror_text(self, event=None):
        if self.after_id:
            self.input_box.after_cancel(self.after_id)

        self.after_id = self.input_box.after(
            500,
            self.correct_text,
        )

    def correct_text(self):
        content = self.input_box.get('1.0', 'end-1c').strip()

        if not content:
            self.update_output('', '')
            return

        messages = [
            {
                'role': 'user',
                'content': content,
            }
        ]

        prompt = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )

        output = self.pipe(
            prompt,
            max_new_tokens=128,
            do_sample=False,
            return_full_text=False,
            pad_token_id=self.tokenizer.eos_token_id,
        )

        corrected_text = output[0]['generated_text'].strip()

        self.update_output(
            content,
            corrected_text,
        )

    def update_output(self, original_text, corrected_text):
        self.output_box.configure(state='normal')
        self.output_box.delete('1.0', 'end')

        original_words = original_text.split()
        corrected_words = corrected_text.split()

        matcher = difflib.SequenceMatcher(
            None,
            original_words,
            corrected_words,
        )

        output_words = []
        styled_words = []

        for operation, _, _, corrected_start, corrected_end in matcher.get_opcodes():
            words = corrected_words[corrected_start:corrected_end]

            output_words.extend(words)
            styled_words.extend(
                [operation != 'equal'] * len(words)
            )

        for index, (word, styled) in enumerate(zip(output_words, styled_words)):
            if index > 0:
                self.output_box.insert('end', ' ')

            start = self.output_box.index('end-1c')
            self.output_box.insert('end', word)
            end = self.output_box.index('end-1c')

            if styled:
                self.output_box.tag_add(
                    'corrected',
                    start,
                    end,
                )

        self.output_box.configure(state='disabled')

    def main_view(self):
        ctk.set_appearance_mode('System')
        ctk.set_default_color_theme('blue')

        app = ctk.CTk()
        app.title('Spelling Correction')
        app.geometry('520x420')

        title = ctk.CTkLabel(
            app,
            text='Spelling Correction',
            font=ctk.CTkFont(
                size=20,
                weight='bold',
            ),
        )
        title.pack(pady=(16, 8))

        self.output_box = ctk.CTkTextbox(
            app,
            width=480,
            height=120,
            wrap='word',
        )
        self.output_box.pack(
            padx=16,
            pady=(0, 12),
            fill='both',
            expand=False,
        )
        self.output_box.configure(state='disabled')
        self.output_box.tag_config(
            'corrected',
            underline=1,
            foreground='red',
        )

        self.input_box = ctk.CTkTextbox(
            app,
            width=480,
            height=160,
            wrap='word',
        )
        self.input_box.pack(
            padx=16,
            pady=(0, 16),
            fill='both',
            expand=True,
        )

        self.input_box.bind(
            '<KeyRelease>',
            self.mirror_text,
        )
        self.input_box.bind(
            '<<Paste>>',
            self.mirror_text,
        )

        app.mainloop()


if __name__ == '__main__':
    gui = SpellingGUI()
    gui.main_view()
