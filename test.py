import webcolors

# Defines Hue ranges for different colors
color_ranges = {
    '0': 'RED',
    '15': 'RED',
    '50': 'ORANGE',
    '66': 'YELLOW',
    '155': 'GREEN',
    '265': 'BLUE',
    '299': 'PURPLE',
    '360': 'RED'
}


def get_hsv(color, color_ranges):
    """
    Converts RGB to HSV value
    :param color: takes in a hex code
    :return: HSV value
    """
    
    # Calculates HSV values
    r0, g0, b0 = webcolors.hex_to_rgb(color)
    r = r0 / 255
    g = g0 / 255
    b = b0 / 255
    c_max = max(r, g, b)
    c_min = min(r, g, b)
    delta = c_max - c_min

    h = 0
    s = 0
    v = 0
    if delta == 0:
        h = 0
    elif c_max == r:
        h = 60 * (((g - b) / delta) % 6)
    elif c_max == g:
        h = 60 * (((b - r) / delta) + 2)
    elif c_max == b:
        h = 60 * (((r - g) / delta) + 4)

    if c_max == 0:
        s = 0
    else:
        s = delta / c_max

    v = c_max
    
    # !Checking if it matches any color. This is really primitive and inefficient right now. There's probably some Python method that could shorten it.
    # !color_list: holds a 2-D array out of the dictionary. There's definitely a better way to do this
    color_list = []
    for key, item in color_ranges.items():
        color_list.append([key, item])
    
    # !Checks if the Hue value is between two Hue numbers from the list 
    for iteration in range(0,len(color_list)-1):
        if h > int(color_list[iteration][0]) and h <=int(color_list[iteration + 1][0]):
            # !I made it print the name of the color for testing purposes
            print(color_list[iteration + 1][1])
    return h, s, v

# !This tests it. You can change the input color and check if it outputs the correct color
get_hsv('#ffd27a', color_ranges)

#!!! Primary and secondary branches