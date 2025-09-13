class NightLight:
    def __init__(self):
        self.running = False

    def start(self):
        """Enable night light / gamma ramp"""
        if not self.running:
            self.running = True
            # TODO: Replace with your gamma ramp / night light code
            print("[NightLight] Enabled")

    def stop(self):
        """Disable night light / gamma ramp"""
        if self.running:
            self.running = False
            # TODO: Reset gamma ramp
            print("[NightLight] Disabled")
