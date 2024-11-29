import math
import numpy as np
from PIL import Image, ImageFilter


def _closestDivisors(n):  # https://stackoverflow.com/questions/40700302/find-the-two-highest-factors-of-a-single-number-that-are-closest-to-each-other
    a = round(math.sqrt(n))
    while n % a > 0: a -= 1
    return a, n//a

def colour_string(words:list, rgb_values:np.ndarray):
    punctuations = ['.', ',', '?', '!', ':', ';']
    rgb_word_string = ''
    for word, rgb in zip(words, rgb_values):
        # print(word)
        r,g,b = rgb.ravel()
       
        rgb_color_code = f"\033[38;2;{r};{g};{b}m"  # ANSI escape code for RGB text
        reset_code = "\033[0m" # Reset ANSI code
        
        next_word = None if word == words[-1] else words[words.index(word)+1]  # don't add space after word if followed by e.g. ?
        space= '' if next_word in punctuations else ' '
        if space == '':
            print(f"{word, next_word}")
        # if word in punctuations:
        #     print(f"{word, next_word}")
            
        rgb_word_string += f"{rgb_color_code}{word.strip()}{reset_code}{space}"
    return rgb_word_string

def data_to_image(rgb_data:np.ndarray, out_path='', blur_strength=0):
    width, height = _closestDivisors(len(rgb_data))
    # Reshape the array into a grid (height x width x 3)
    rgb_grid = rgb_data.reshape((height, width, 3))
    rgb_grid = rgb_grid.astype(np.uint8)
    # Create the image using PIL
    image = Image.fromarray(rgb_grid, 'RGB')
    blurred_image = image.filter(ImageFilter.BoxBlur(blur_strength))
    if out_path:
        blurred_image.save(out_path)
        print(f"Image saved as {out_path}")
    else:
        blurred_image.show()