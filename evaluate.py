import fine_tuning as ft
import LwF2 as lwf
import iCaRL2 as icarl

import utils


def main():
    for i in ['1','2','3']:
        acc_lwf = lwf.train(i)
        #acc_icarl = icarl.train(i)
    print(acc_lwf)
















if __name__ == '__main__':
    main()
