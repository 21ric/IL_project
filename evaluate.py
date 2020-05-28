import fine_tuning as ft
import LwF2 as lwf
import iCaRL2 as icarl

import utils


def main():
    for i in ['1','2','3']:
        for inc_lrng in [ft,lwf,icarl]:
        
           print(f"Incremental learning: {str(inc_lrng)}\n")
           print(f"Classes group {i}\n")
           acc_ = inc_lrng.incremental_learning(i)
           #acc_icarl = icarl.train(i)

           print(acc_)
















if __name__ == '__main__':
    main()
