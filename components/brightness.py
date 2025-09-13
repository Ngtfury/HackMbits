import time
import screen_brightness_control as sbc

def subtle_blink_pulse(pulse_percent=5, pulse_duration=0.5):
    try:
        current_brightness = sbc.get_brightness(display=0)[0]
        new_brightness = max(0, min(100, current_brightness + pulse_percent))
        sbc.set_brightness(new_brightness)
        time.sleep(pulse_duration)
        sbc.set_brightness(current_brightness)
    except Exception as e:
        print("Error changing brightness:", e)


# Example usage: pulse subtly every few seconds
if __name__ == "__main__":
    for _ in range(3):  # repeat 3 times for demo
        subtle_blink_pulse()
        time.sleep(3)  # wait 3 seconds between pulses
