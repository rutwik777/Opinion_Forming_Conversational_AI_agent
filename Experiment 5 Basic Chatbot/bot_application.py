from pydoc import plain
from tkinter import *
from bot_v2 import get_response, bot_name
#https://github.com/python-engineer/pytorch-chatbot
#Set the colors for UI
BKG_GREY = "#ABB2B9"
BG_COLOR = "#17202A"
TEXT_COLOR = "#EAECEE"

#Set the font for UI
FONT = "Times 16"
FONT_BOLD = "Times 16 bold"

class ChatBotApplication:
    def __init__(self):
        self.window = Tk()
        self._setup_main_window()

    def _setup_main_window(self):
        self.window.title("Conversational AI on Economics")
        self.window.resizable(height=True, width=True)
        self.window.configure(width=550, height=600, bg=BG_COLOR)

        head_label = Label(self.window, bg=BG_COLOR, fg=TEXT_COLOR, text="Namaste", font=FONT_BOLD, pady=10) #Head label definition
        head_label.place(relwidth=1) 

        line = Label(self.window, width=450, bg=BG_COLOR)
        line.place(relwidth=1, rely=0.07, relheight=0.012)

        self.text_widget = Text(self.window, width=20, height=2, bg=BG_COLOR, fg=TEXT_COLOR, font=FONT, padx=5, pady=5)
        self.text_widget.place(relheight=0.75, relwidth=1, rely=0.08)
        self.text_widget.configure(cursor="arrow", state=DISABLED)

        scrollbar = Scrollbar(self.text_widget)
        scrollbar.place(relheight=1, relx=0.975)
        scrollbar.configure(command=self.text_widget.yview)

        bottomlabel = Label(self.window, bg=BKG_GREY, height=80)
        bottomlabel.place(relwidth=1, rely=0.8)

        self.msg_entry = Entry(bottomlabel, bg="GREEN", fg=TEXT_COLOR, font=FONT)
        self.msg_entry.place(relwidth=0.75, relheight=0.05, rely=0.010, relx=0.011)
        self.msg_entry.focus()
        self.msg_entry.bind("<Return>", self._on_hit_enter)

        send_button = Button(bottomlabel, text="Send", font=FONT_BOLD, width=20, bg=BKG_GREY,
                             command=lambda: self._on_enter_pressed(None))
        send_button.place(relx=0.77, rely=0.008, relheight=0.06, relwidth=0.22)

    def _on_hit_enter(self, event):
        msg = self.msg_entry.get()
        self._message_display(msg, "You")

    def _message_display(self, msg, sender):
        self.msg_entry.delete(0, END)
        msg1 = f"{sender}: {msg}\n\n"
        self.text_widget.configure(state=NORMAL)
        self.text_widget.insert(END, msg1)
        self.text_widget.configure(state=DISABLED)
        
        msg2 = f"{bot_name}: {get_response(msg)}\n\n"
        self.text_widget.configure(state=NORMAL)
        self.text_widget.insert(END, msg2)
        self.text_widget.configure(state=DISABLED)
        
        self.text_widget.see(END)

    def run(self):
        self.window.mainloop()

if __name__ == "__main__":
    app = ChatBotApplication()
    app.run()