import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'    # Disable debugging and warning logs
import numpy as np

from src.mstar_frame import mstar_frame

from src.cvnn_model import cvnn_model
from src.cnn_model import cnn_model
from src.cnn_ensemble import cnn_ensemble
from src.dnn_model import dnn_model

from utils.plot import plot


def main():

    # Unpack MSTAR dataframe.
    mdf = mstar_frame(ctype='all')  # Both polar and cartesian complex formats

    train_df_iq, train_df_mp, train_id, test_df_iq, test_df_mp, test_id = mdf.get_mstar_data(plot_flag=False)

    # Define complex-valued model.
    cvnn = cvnn_model(train_df=train_df_iq, train_id=train_id, \
                      test_df=test_df_iq, test_id=test_id, init_flag=True)
    
    # Train complex-valued model.
    if cvnn.init_flag:

        train_loss_cvnn, val_loss_cvnn, train_acc_cvnn, val_acc_cvnn = cvnn.train(plot_flag=True)

        print()
        print('====================')
        print(' Complex-Valued CNN')
        print('====================')
        print()

        print('Train CCE Loss:          %.4f' % train_loss_cvnn)
        print('Validation CCE Loss:     %.4f' % val_loss_cvnn)
        print()

        print('Train Accuracy:          %.4f' % train_acc_cvnn)
        print('Validation Accuracy:     %.4f' % val_acc_cvnn)

    # Evaluate complex-valued model.
    acc_cvnn, prc_cvnn, rcl_cvnn, f1_cvnn = cvnn.evaluate(cvnn.test_df, cvnn.test_id_oh, plot_flag=True)

    print()
    print('====================')
    print(' Complex-Valued CNN')
    print('====================')
    print()

    print('Accuracy Score:          %.4f' % acc_cvnn)
    print('Precision Score:         %.4f' % prc_cvnn)
    print('Recall Score:            %.4f' % rcl_cvnn)
    print('F1 Score:                %.4f' % f1_cvnn)

    # Define real-valued models for cartesian complex format.
    cnn_i = cnn_model(train_df=train_df_iq, train_id=train_id, \
                      test_df=test_df_iq, test_id=test_id, ctype='i', init_flag=False)  # In-phase component
    cnn_q = cnn_model(train_df=train_df_iq, train_id=train_id, \
                      test_df=test_df_iq, test_id=test_id, ctype='q', init_flag=False)  # Quadrature component

    # Define real-valued models for polar complex format.
    cnn_m = cnn_model(train_df=train_df_mp, train_id=train_id, \
                      filter_size=48, kernel_size=5, learning_rate=0.001, dropout_rate=0.5, \
                      test_df=test_df_mp, test_id=test_id, ctype='m', init_flag=False)   # Magnitude component
    cnn_p = cnn_model(train_df=train_df_mp, train_id=train_id, \
                      filter_size=24, kernel_size=3, learning_rate=0.001, dropout_rate=0.5, \
                      test_df=test_df_mp, test_id=test_id, ctype='p', init_flag=False)   # Phase component
    
    # Train real-valued model for in-phase component.
    if cnn_i.init_flag:

        train_loss_cnn_i, val_loss_cnn_i, train_acc_cnn_i, val_acc_cnn_i = cnn_i.train(patience=5, batch_size=64, plot_flag=True)

        print()
        print('============================')
        print(' Real-Valued CNN (In-Phase)')
        print('============================')
        print()

        print('Train CCE Loss:          %.4f' % train_loss_cnn_i)
        print('Validation CCE Loss:     %.4f' % val_loss_cnn_i)
        print()

        print('Train Accuracy:          %.4f' % train_acc_cnn_i)
        print('Validation Accuracy:     %.4f' % val_acc_cnn_i)
    
    # Train real-valued model for quadrature component.
    if cnn_q.init_flag:

        train_loss_cnn_q, val_loss_cnn_q, train_acc_cnn_q, val_acc_cnn_q = cnn_q.train(plot_flag=True)

        print()
        print('==============================')
        print(' Real-Valued CNN (Quadrature)')
        print('==============================')
        print()

        print('Train CCE Loss:          %.4f' % train_loss_cnn_q)
        print('Validation CCE Loss:     %.4f' % val_loss_cnn_q)
        print()

        print('Train Accuracy:          %.4f' % train_acc_cnn_q)
        print('Validation Accuracy:     %.4f' % val_acc_cnn_q)

    # Train real-valued model for phase component.
    if cnn_p.init_flag:

        train_loss_cnn_p, val_loss_cnn_p, train_acc_cnn_p, val_acc_cnn_p = cnn_p.train(plot_flag=False)

        print()
        print('=========================')
        print(' Real-Valued CNN (Phase)')
        print('=========================')
        print()

        print('Train CCE Loss:          %.4f' % train_loss_cnn_p)
        print('Validation CCE Loss:     %.4f' % val_loss_cnn_p)
        print()

        print('Train Accuracy:          %.4f' % train_acc_cnn_p)
        print('Validation Accuracy:     %.4f' % val_acc_cnn_p)

    # Train real-valued model for magnitude component.
    if cnn_m.init_flag:

        train_loss_cnn_m, val_loss_cnn_m, train_acc_cnn_m, val_acc_cnn_m = cnn_m.train(plot_flag=False)

        print()
        print('=============================')
        print(' Real-Valued CNN (Magnitude)')
        print('=============================')
        print()

        print('Train CCE Loss:          %.4f' % train_loss_cnn_m)
        print('Validation CCE Loss:     %.4f' % val_loss_cnn_m)
        print()

        print('Train Accuracy:          %.4f' % train_acc_cnn_m)
        print('Validation Accuracy:     %.4f' % val_acc_cnn_m)

    # Evaluate real-valued model for magnitude component (polar).
    acc_cnn_m, prc_cnn_m, rcl_cnn_m, f1_cnn_m = cnn_m.evaluate(cnn_m.test_df, cnn_m.test_id_oh, plot_flag=False)

    print()
    print('=============================')
    print(' Real-Valued CNN (Magnitude)')
    print('=============================')
    print()

    print('Accuracy Score:          %.4f' % acc_cnn_m)
    print('Precision Score:         %.4f' % prc_cnn_m)
    print('Recall Score:            %.4f' % rcl_cnn_m)
    print('F1 Score:                %.4f' % f1_cnn_m)
    print()

    # Define real-valued ensemble.
    ensemble_iq = cnn_ensemble(train_df=train_df_iq, train_id=train_id, \
                               test_df=test_df_iq, test_id=test_id, \
                               init_flag=False, train_flag=False)
    
    # ensemble_mp = cnn_ensemble(train_df=train_df_mp, train_id=train_id, \
    #                            test_df=test_df_mp, test_id=test_id, \
    #                            pol_flag=True, init_flag=False, train_flag=True)
    
    # Evaluate real-valued ensemble for cartesian complex format.
    acc_ens_iq_w, prc_ens_iq_w, rcl_ens_iq_w, f1_ens_iq_w = ensemble_iq.evaluate(ensemble_iq.test_df, ensemble_iq.test_id_oh, w=ensemble_iq.opt_wts, plot_flag=True)
    acc_ens_iq_a, prc_ens_iq_a, rcl_ens_iq_a, f1_ens_iq_a = ensemble_iq.evaluate(ensemble_iq.test_df, ensemble_iq.test_id_oh, w=ensemble_iq.eq_wts)
    
    print()
    print('=======================================')
    print(' Weighted-Average Ensemble (Cartesian)')
    print('=======================================')
    print()

    print('Accuracy Score:      %.4f' % acc_ens_iq_w)
    print('Precision Score:     %.4f' % prc_ens_iq_w)
    print('Recall Score:        %.4f' % rcl_ens_iq_w)
    print('F1 Score:            %.4f' % f1_ens_iq_w)
    
    print()
    print('================================')
    print(' Averaging Ensemble (Cartesian)')
    print('================================')
    print()

    print('Accuracy Score:      %.4f' % acc_ens_iq_a)
    print('Precision Score:     %.4f' % prc_ens_iq_a)
    print('Recall Score:        %.4f' % rcl_ens_iq_a)
    print('F1 Score:            %.4f' % f1_ens_iq_a)

    # Evaluate real-valued ensemble for polar complex format.
    acc_ens_mp_w, prc_ens_mp_w, rcl_ens_mp_w, f1_ens_mp_w = ensemble_mp.evaluate(ensemble_mp.test_df, ensemble_mp.test_id_oh, w=ensemble_mp.opt_wts, plot_flag=True)
    acc_ens_mp_a, prc_ens_mp_a, rcl_ens_mp_a, f1_ens_mp_a = ensemble_mp.evaluate(ensemble_mp.test_df, ensemble_mp.test_id_oh, w=ensemble_mp.eq_wts)
    
    print()
    print('===================================')
    print(' Weighted-Average Ensemble (Polar)')
    print('===================================')
    print()

    print('Accuracy Score:      %.4f' % acc_ens_mp_w)
    print('Precision Score:     %.4f' % prc_ens_mp_w)
    print('Recall Score:        %.4f' % rcl_ens_mp_w)
    print('F1 Score:            %.4f' % f1_ens_mp_w)
    
    print()
    print('============================')
    print(' Averaging Ensemble (Polar)')
    print('============================')
    print()

    print('Accuracy Score:      %.4f' % acc_ens_mp_a)
    print('Precision Score:     %.4f' % prc_ens_mp_a)
    print('Recall Score:        %.4f' % rcl_ens_mp_a)
    print('F1 Score:            %.4f' % f1_ens_mp_a)
    print()

    # Train real-valued model for in-phase component.
    if cnn_i.init_flag:

        train_loss_cnn_i, val_loss_cnn_i, train_acc_cnn_i, val_acc_cnn_i = cnn_i.train(patience=5, batch_size=64, plot_flag=True)

        print()
        print('============================')
        print(' Real-Valued CNN (In-Phase)')
        print('============================')
        print()

        print('Train CCE Loss:          %.4f' % train_loss_cnn_i)
        print('Validation CCE Loss:     %.4f' % val_loss_cnn_i)
        print()

        print('Train Accuracy:          %.4f' % train_acc_cnn_i)
        print('Validation Accuracy:     %.4f' % val_acc_cnn_i)


if __name__ == '__main__':

    main()

