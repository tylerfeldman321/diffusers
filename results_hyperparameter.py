import os

def main():
    """
        Added to automatically output results with different text guided hyperparameter
    """
    TOTAL_IMAGES = 10
    PROMPT_PER_IMAGE = 3
    MASK_HYPERPARAMETER = 0.2
    MASK_FREQUENCY = 10

    text_hyperparameter = [1, 3, 5, 6.5, 7.5, 8.5, 10]

    for param in text_hyperparameter:
        dir = f"results/text_guidance={param}/"
        command = "python results_generation.py -t " + str(param) + " -m 0.2 -f 10 -o " + str(dir)
        print(f"STARTED RUNNING THE RESULTS WITH TEXT HYPERPARAMETER {param}")
        os.system(command)
        print(f"FINISHED RUNNING THE RESULTS WITH TEXT HYPERPARAMETER {param}")
    print("--------Done----------")

if __name__=="__main__":
    main()