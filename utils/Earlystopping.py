import math

class EarlyStopping:
    def __init__(self, patience=5, verbose=False, mode="min"):
        self.patience = patience
        self.verbose = verbose
        self.mode = mode
        self.patience_cnt = 0
        self.earlystop = False
        self.best_epoch = False
        
        if self.mode == "min":
            self.ref = math.inf
        elif self.mode == "max":
            self.ref = -math.inf          
        else:
            raise ValueError("mode can be 'min' or 'max' only.")

    def __call__(self, cur_ref):
        if cur_ref < 0:
            return
        if (self.mode == "max" and cur_ref > self.ref) \
            or (self.mode == "min" and cur_ref < self.ref):      
                if self.verbose:
                    print(f'Earlystop: the best target value is changed. [{self.ref:.4f} > {cur_ref:.4f}]')
                self.patience_cnt = 0
                self.ref = cur_ref
                self.best_epoch = True
        else:
            self.patience_cnt += 1
            self.best_epoch = False
            if self.patience_cnt >= self.patience:
                if self.verbose:
                    print('earlystopping')   
                self.earlystop = True
