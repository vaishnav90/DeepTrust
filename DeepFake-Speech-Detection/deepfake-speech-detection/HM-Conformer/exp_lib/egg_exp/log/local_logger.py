import os
import torch
import shutil

from ._interface import Logger

PATH = None

class LocalLogger(Logger):
    """Save experiment logs to local storage.
    """
    def __init__(self, path, description=None, scripts=None):
        # If the target directory already exists, NEVER delete it.
        # Instead, create a new unique run directory to avoid wiping checkpoints/results
        # from previous runs.
        #
        # Previous behavior:
        #   if os.path.exists(path): shutil.rmtree(path)
        #
        # This was dangerous when users re-ran an experiment pointing to an existing
        # results directory (common in notebooks).
        run_path = path
        if os.path.exists(run_path):
            import datetime
            ts = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
            # Create a unique suffix if multiple runs start within the same second.
            candidate = f"{path}-run-{ts}"
            idx = 1
            while os.path.exists(candidate):
                idx += 1
                candidate = f"{path}-run-{ts}-{idx}"
            run_path = candidate

        # set path
        self.path = run_path

        # make directory
        os.makedirs(self.path, exist_ok=True)

        # save description
        if description is not None:
            self.log_text('description', description)

        # script backup
        if scripts is not None:
            shutil.copytree(scripts, f'{self.path}/scripts')
            
            for root, dirs, _ in os.walk(f'{self.path}/scripts'):
                for dir in dirs:
                    if dir == '__pycache__':
                        shutil.rmtree(f'{root}/{dir}')

    def log_text(self, name, text):
        f = f'{self.path}/{name}.txt'
        os.makedirs(os.path.dirname(f), exist_ok=True)
        mode = 'a' if os.path.exists(f) else 'w'
        f = open(f, mode, encoding='utf-8')
        f.write(text)
        f.close()
        
    def log_arguments(self, arguments):
        for k, v in arguments.items():
            self.log_text('parameters', f'{k}: {v}\n')

    def log_metric(self, name, value, step=None):
        if step is None:
            msg = f'{name}: {value}\n'
        else:
            msg = f'[{step}] {name}: {value}\n'

        self.log_text(name, msg)

    def log_image(self, name, image):
        pass
        # f = f'{self.path}/{name}.png'
        # os.makedirs(os.path.dirname(f), exist_ok=True)
        # image.save(f, 'PNG')
        
    def save_model(self, name, state_dict):
        # set model's results path
        path = os.path.join(self.path, "models")
        
        # make directory
        if not os.path.exists(path):
            os.makedirs(path)
        
        f = f'{path}/{name}.pt'
        torch.save(state_dict, f)