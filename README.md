
# Set Up

please go through the checklist below for all the dependencies set up.

## A. Hardware

highly recommend running RL with Cuda on Windows OS.

## B. Software

1. `Github Desktop` (Please also register an account on Github).
2. `Visual Studio Code 2023`.
3. `Rhino 7`.
4. `Miniconda`.
5. `Cuda Toolkit` (Cuda is Nvidia GPU only).
   
## C. Virtual Env

install a virtual environment in anaconda, steps as follows:
### for windows
1. fire up `Anaconda Prompt`
2. type in:
   
        conda create -n RL -c anaconda python=3.9 tensorflow-gpu==2.6.0 matplotlib pandas openpyxl nb_conda

### For Mac

https://developer.apple.com/metal/tensorflow-plugin/

      python3 -m venv ~/venv-RL
      source ~/venv-RL/bin/activate
      python -m pip install -U pip
      python -m pip install tensorflow matplotlib pandas openpyxl

#### For M1 MAC run also
      python -m pip install tensorflow-metal
      
#### To remove the virtual environment:      

      sudo rm -rf ~/venv-RL

# Training RL

Once all set, go to the `Anaconda Prompt`:

      cd Path/to/RL-FabDesign
      conda activate RL
      python Train_RL.py

and waiting for RL to have fun playing with your project!
This results in 100 cases exported to throughout the training process. result 1 is less learned than result 100

# Visualize in Rhino

1. Open `Design.3dm` and `Design.gh`
2. Setup the path to excel file.
3. use slider to pick result to visualize.


# Inference in Rhino

TBD

# Customize

once customized your own project, follow the checklist below before running the RL training.

## A. Design Library

1. check the modified 3D tiles in `Design.3dm` and `Design.gh`, and make sure tile numbers matches n_actions, canvas scale matches all_canvas, room types matches Num_roomTypes, all defined in `Train_RL.py`.
2. check `Team_Project` folder files `adjacent_rules.xlxs` and `fabrication_type_list.xlxs`, ensuring they match the tile designs in `Design.3dm`.

## B. Goals

1. check `Team_Project` folder file `room_area_goal.csv`, or your customized goals.
2. check the fabrication time goal specified in `Train_RL.py`.

## C. Training Settings

go to the `Train_RL.py` file, and check as follows:
1. the ind_stair refers to the staircase tiles in `Design.3dm` matches accurately.
2. switching between observe global and observe local requires changes in input_dims and ENV.step.
3. define n_games = 100 when testing, and n_games = 10000 or more when training.


