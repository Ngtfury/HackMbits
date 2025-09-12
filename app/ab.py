import screen_brightness_control as sbc

# Get current brightness
print(sbc.get_brightness())

# Set brightness (0-100)
sbc.set_brightness(50)
screen_brightness = get_screen_brightness()
print(f'Screen brightness: {screen_brightness}')
# Increase/decrease brightness
sbc.set_brightness('+10')  # +10%
screen_brightness = get_screen_brightness()
print(f'Screen brightness: {screen_brightness}')
sbc.set_brightness('-10')  # -10%
screen_brightness = get_screen_brightness()
print(f'Screen brightness: {screen_brightness}')
