import pyttsx3
import threading
import queue
import time

class Speaker:
    def __init__(self):
        self.engine = pyttsx3.init()
        self.engine.setProperty("rate", 170)

        self.queue = queue.Queue()
        self.running = True

        thread = threading.Thread(target=self._run, daemon=True)
        thread.start()

    def say(self, text):
        """Queue new speech unless duplicate"""
        try:
            last = self.queue.queue[-1] if not self.queue.empty() else ""
        except:
            last = ""

        if text != last:
            self.queue.put(text)

    def _run(self):
        """Background speaker thread"""
        while self.running:
            if not self.queue.empty():
                text = self.queue.get()
                self.engine.say(text)
                self.engine.runAndWait()
            time.sleep(0.05)

    def stop(self):
        self.running = False
