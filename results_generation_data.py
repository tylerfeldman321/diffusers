import os


def insert_at_end_of_path(path, addition):
    extsplit = os.path.splitext(path)
    return extsplit[0] + '_' + addition + extsplit[1]


INPUT_IMAGE = ['apple_man.jpg',
               'dog_on_bench.png',
               'flowers.jpg',
               'lebron.jpeg',
               'london.jpg',
               'statue.jpeg',
               'sunflower.jpg',
               'tree_and_building.png',
               'two_men_on_bench.png',
               'woman_and_fence.png']

INPUT_MASK = [insert_at_end_of_path(image, 'mask') for image in INPUT_IMAGE]

COMMANDS_PER_IMAGE = 3
COMMAND = ['Turn the apple into an orange', 'Make the apple a flower', 'Have a bite taken out of the apple',
           'Turn the dog black', 'Make the dog a Walker Hound', 'Remove the dog',
           'Replace the flowers with roses', 'Make the flowers covered in snow', 'Shrink the flowers',
           'Turn the basketball to a pumpkin', 'Make the basketball blue', 'Remove the basketball',
           'Add a mountain to the background', 'Add fireworks to the sky', 'Make the sky cloudy',
           'Give him clown makeup', 'Give him black hair', 'Make him bald',
           'Replace the sunflower with a tulip', 'Give the sunflower a human face', 'Give the sunflower sunglasses',
           'Change the tree to a cactus', 'Remove the tree', 'Make the tree blue',
           'Give them sandals', 'Make them barefoot', 'Give them cleats',
           'Make her blonde', 'Make her hair straight', 'Make her hair shorter']

OUTPUT_IMAGE_ALIASES = ['orange', 'flower', 'bite',
                        'black', 'hound', 'remove',
                        'roses', 'snow', 'shrink',
                        'pumpkin', 'blue', 'remove',
                        'mountain', 'fireworks', 'cloudy',
                        'clown', 'black', 'bald',
                        'tulip', 'human_face', 'sunglasses',
                        'cactus', 'remove', 'blue',
                        'sandals', 'barefoot', 'cleats',
                        'blonde', 'straight', 'short']


OUTPUT_IMAGE = [insert_at_end_of_path(INPUT_IMAGE[i // COMMANDS_PER_IMAGE], alias + '_{}_{}_{}') for i, alias in enumerate(OUTPUT_IMAGE_ALIASES)]
VANILLA_OUTPUT = [insert_at_end_of_path(INPUT_IMAGE[i // COMMANDS_PER_IMAGE], alias + '_vanilla') for i, alias in enumerate(OUTPUT_IMAGE_ALIASES)]

INPUT_DIR = 'inputs/'
OUTPUT_DIR = 'results/'
