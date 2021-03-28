# Low shot learning with imprinted weights and fine tuning improvments (on CIFAR100)
## Achieving 80% accuracy on Imprinted (Low-Shot) class with 5 shots or more on CIFAR100
Implementation of <a href = 'https://openaccess.thecvf.com/content_cvpr_2018/papers/Qi_Low-Shot_Learning_With_CVPR_2018_paper.pdf'> 
Low-Shot Learning With Imprinted Weights </a> with contributions on CIFAR100.

## Differences and contributions
- Using Resnet18 + SGD
- Using adapative and weighted loss function for FineTuning
- Using special training phase to better perform FineTuning
- Adaptative code for CIFAR100 and other datasets

## Explanations
Code has been made for CIFAR100, but other Data can be used easily. YAML config will ensure reproducibility of the resuts. **main.py** will provide both imprinted model and imprinted+finetuned model for a class chosen in the config file. Note that all the code is built on several iterations over number of shots and chosen classes for imprinting to get statistically relevant performances. Indeed, performances depend on two main factors:
- The class chosen for imprinting weights. The more the class is discrimant compared to the initial class learnt my the model, the more imprinting and fine-tuning will be efficient.
- The instance(s) of the Low-Shot class. Some instances of the dataset present more salliant points and will improve model's performances.

Thus, face to this potential sources of variability, iterating over instances (sampled randomly at each iterations) and classes (sampled randomly too in case you are using the main_full_exp.py to validate the entire chain) will act like a "Monte Carlo" process to get statistically reliable values. The adapative loss (weighted according to the configuration of the model) and training (freezing weights half the time for initial classes and preveting from over-fitting on fine tining controlling hyperparameters) added here (see code), are also tackling this issue and with 5 shots or more, the model can have as good performances (**over 80%**) for imprinted class than for initial classes with low standart devations (see **Results** section).
The adaptive loss and training phase has been added and developped here as Fine Tuning did not gave significant improvement in many cases as the model sometimes overfits on initial classes especially when the imprinted class is not discrimant enough. The contributions made always improved the results after fine tuning.

## Usage
- **cd**  ./path/to/the/repository
- Create your config file in yaml from ./config/**config.py**
- Do not forget to generate base model (if you do not have already one) to perform imprinting on. To do so set CreateNet boolean to **True** in your config.yaml file, when model is created you can put it back to **False**
- Launch **main.py** to perform imprinting on one class
- Launch **main_full_exp.py** to launch full experience (on various classes and several iterations)
- Results will be saved in ./Results folder with a .txt file with in order dict containing (mean of accuracies obtained over iterations, 
all_accuracies per category, standart deviation per category and confidence interval (95%) for mean accuracies). Each experience's result is saved in 
a folder named with a timestamp and with the associated config file to reproduce results. This folder will also contain imprinted models with the following name model_ or model_ft_ (with finetuning) + number-of-shots_iteration.pth.
Feel free to use any other datasets with the code. You might have to reconsider adapative loss and training developped in the code.

## Results
The following results are extracted from an experience made on a network trained on 40 inital classes and one more classes imprinted (41 classes in total). Experiences has been made on different classes (sampled randomly) and different instances (sampled randomply too) at each iteration (with obviously the initial network reseted to initial weights at each experiences) to obtain relevant results. 
| num_shots  |  1 | 5  |  10 | 20  |
|---|---|---|---|---|
| Imprinted class accuracy (%) | 12.6  | 33.1  |  43.4| 77.56  |
| Imprinted class accuracy (%) with Fine Funing| **29.5**  | **80.8**  |  **81.4** | **80.8**  |

All accuracy here are more or less 2,5% (confidence interval at 95%). Global accurat of the net (all classes) is aroung 84%.


## References
 - [1]: H. Qi, M. Brown and D. Lowe. "Low-Shot Learning with Imprinted Weights", in CVPR, 2018.
 - [2]: YU1ut/Imprinted  on <a href = 'https://github.com/YU1ut/imprinted-weights'> GitHub </a>
