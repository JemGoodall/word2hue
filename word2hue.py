from argparse import ArgumentParser
from embeddings_to_rgb import embeddings_to_rgb
from DimensionalityReducer import REDUCTION_METHODS
from display import colour_string, data_to_image
from text_processing import NLTK_TEXT_CHOICES
import os
import sys

def main():
    a = ArgumentParser()
    group = a.add_mutually_exclusive_group(required=True)
    group.add_argument('-t', '--text', type=str, choices=NLTK_TEXT_CHOICES, help="Name of nltk text to use")
    group.add_argument('-i', '--input_txt_file', type=str, help='Path to .txt file containing text to process.')
    
    a.add_argument('-n', '--n_tokens', default=-1, type=int, help='number of tokens to use from given text')
    a.add_argument('-s', '--save', action='store_true', help='Defualt save to plots/word2hue_{booktitle|text}.png. If not present and no --out_path provided, will display plot."')
    a.add_argument('-o', '--out_path', help="Specify save location of image. --save automatically set to True if this flag is used." )
    a.add_argument('-d', '--display_text', action='store_true', help='Bool. If present, will print coloured text to terminal')
    a.add_argument('-p', '--punctuation', action='store_true', help='Bool. If present, will keep punctuation in the text. Default is to strip it.')
    a.add_argument('-r', '--reduction_method', default='pca', choices=REDUCTION_METHODS, help='Specify reduction method.')
    a.add_argument('--blur', default=0, type=int, help='Define blur strength. Higher --> more blur. 0 = no blur applied.') 
    args = a.parse_args()
    gutenberg_text, input_txt_file, n_tokens, save, out_path, display_text, punctuation, reduction_method, blur = vars(args).values()

    text = gutenberg_text if gutenberg_text else input_txt_file
    save_location = out_path
    if save and not out_path:
        save_location = f'plots/{os.path.splitext(text)[0]}.png'
    
    tokens, word_rgb_values = embeddings_to_rgb(text, n_tokens, save_location, punctuation, reduction_method)
    if display_text:
        print(colour_string(tokens, word_rgb_values))

    if save or out_path:
        data_to_image(word_rgb_values, out_path=save_location, blur_strength=blur)

if __name__ == '__main__':
    main()