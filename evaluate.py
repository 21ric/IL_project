import fine_tuning as ft
import LwF2 as lwf
import iCaRL2 as icarl

import utils


def main():
    for i, inc_lrng in enumerate([lwf,ft,icarl]):
        
        print(f"Incremental learning: {str(inc_lrng)}\n")
        print(f"Classes group {i}\n")
        acc_lwf = inc_lrng.incremental_learning(str(i))
        #acc_icarl = icarl.train(i)

    print(acc_lwf)
















if __name__ == '__main__':
    main()
