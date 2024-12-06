import torch
from model import pde
import numpy as np
import time
from utils import read_csv, write_csv
from pathlib import Path


def train(
    nb_epoch,
    train_loss,
    test_loss,
    weight_data_init,
    weight_pde_init,
    weight_border_init,
    dynamic_weights,
    lr_weights,
    model,
    loss,
    optimizer,
    X_train,
    U_train,
    X_test_pde,
    X_test_data,
    U_test_data,
    X_pde,
    Re,
    time_start,
    f,
    folder_result,
    save_rate,
    batch_size,
    scheduler,
    X_border,
    X_border_test,
    mean_std,
    ya0,
    w_0,
    param_adim
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    nb_it_tot = nb_epoch + len(train_loss["total"])
    print(
        f"--------------------------\nStarting at epoch: {len(train_loss['total'])}"
        + "\n--------------------------"
    )
    print(
        f"--------------------------\nStarting at epoch: {len(train_loss['total'])}\n------------"
        + "--------------",
        file=f,
    )
    weight_border = weight_border_init
    weight_data = weight_data_init
    weight_pde = weight_pde_init

    for epoch in range(len(train_loss["total"]), nb_it_tot):
        loss_batch_train = {"total": [], "data": [], "pde": [], 'border': []}
        for batch in range(len(X_pde) // batch_size):
            model.train()  # on dit qu'on va entrainer (on a le dropout)
            # loss du pde
            X_pde_batch = X_pde[batch *
                                batch_size: (batch + 1) * batch_size, :]
            pred_pde = model(X_pde_batch)
            pred_pde1, pred_pde2, pred_pde3 = pde(
                pred_pde,
                X_pde_batch,
                Re=Re,
                x_std=mean_std['x_std'],
                y_std=mean_std['y_std'],
                u_mean=mean_std['u_mean'],
                v_mean=mean_std['v_mean'],
                p_std=mean_std['p_std'],
                t_std=mean_std['t_std'],
                u_std=mean_std['u_std'],
                v_std=mean_std['v_std'],
                ya0=ya0,
                w_0=w_0,
                param_adim=param_adim,
                mean_std=mean_std
            )
            loss_pde = (
                torch.mean(pred_pde1**2)
                + torch.mean(pred_pde2**2)
                + torch.mean(pred_pde3**2)
            )

            # loss des points de data
            pred_data = model(X_train)
            loss_data = loss(U_train, pred_data)

            # loss du border
            pred_border = model(X_border)
            goal_border = torch.tensor([-mean_std['u_mean']/mean_std['u_std'], -mean_std['v_mean'] /
                                       mean_std['v_std']], dtype=torch.float32).expand(pred_border.shape[0], 2).to(device)
            loss_border_cylinder = loss(
                pred_border[:, :2], goal_border)  # (MSE)

            loss_totale = weight_data * loss_data + weight_pde * \
                loss_pde + weight_border * loss_border_cylinder

            # Backpropagation
            loss_totale.backward(retain_graph=True)
            optimizer.step()
            optimizer.zero_grad()
            with torch.no_grad():
                loss_batch_train["total"].append(loss_totale.item())
                loss_batch_train["data"].append(loss_data.item())
                loss_batch_train["pde"].append(loss_pde.item())
                loss_batch_train["border"].append(loss_border_cylinder.item())

            if dynamic_weights:
                weight_data_hat += lr_weights * loss_data
                weight_pde_hat += lr_weights * loss_pde
                weight_border_hat += lr_weights * loss_border_cylinder
                sum_weight = weight_data_hat + weight_pde_hat + weight_border_hat
                weight_data = weight_data_hat / sum_weight
                weight_border = weight_border_hat / sum_weight
                weight_pde = weight_pde_hat / sum_weight

        # Pour le test :
        model.eval()

        # loss du pde
        test_pde = model(X_test_pde)
        test_pde1, test_pde2, test_pde3 = pde(
            test_pde,
            X_test_pde,
            Re=Re,
            x_std=mean_std['x_std'],
            y_std=mean_std['y_std'],
            u_mean=mean_std['u_mean'],
            v_mean=mean_std['v_mean'],
            p_std=mean_std['p_std'],
            t_std=mean_std['t_std'],
            u_std=mean_std['u_std'],
            v_std=mean_std['v_std'],
            ya0=ya0,
            w_0=w_0,
            param_adim=param_adim,
            mean_std=mean_std
        )
        loss_test_pde = (
            torch.mean(test_pde1**2)
            + torch.mean(test_pde2**2)
            + torch.mean(test_pde3**2)
        )
        # loss de la data
        test_data = model(X_test_data)
        loss_test_data = loss(U_test_data, test_data)  # (MSE)

        # loss des bords
        pred_border_test = model(X_border_test)
        goal_border_test = torch.tensor([-mean_std['u_mean']/mean_std['u_std'], -mean_std['v_mean'] /
                                        mean_std['v_std']], dtype=torch.float32).expand(pred_border_test.shape[0], 2).to(device)
        loss_test_border = loss(
            pred_border_test[:, :2], goal_border_test)  # (MSE)

        # loss totale
        loss_test = weight_data * loss_test_data + weight_pde * \
            loss_test_pde + weight_border * loss_test_border
        scheduler.step()
        with torch.no_grad():
            test_loss["total"].append(loss_test.item())
            test_loss["data"].append(loss_test_data.item())
            test_loss["pde"].append(loss_test_pde.item())
            test_loss["border"].append(loss_test_border.item())
            train_loss["total"].append(np.mean(loss_batch_train["total"]))
            train_loss["data"].append(np.mean(loss_batch_train["data"]))
            train_loss["pde"].append(np.mean(loss_batch_train["pde"]))
            train_loss["border"].append(np.mean(loss_batch_train["border"]))

        print(f"---------------------\nEpoch {epoch+1}/{nb_it_tot} :")
        print(f"---------------------\nEpoch {epoch+1}/{nb_it_tot} :", file=f)
        print(
            f"Train : loss: {train_loss['total'][-1]:.3e}, data: {train_loss['data'][-1]:.3e}, pde: {train_loss['pde'][-1]:.3e}, border: {train_loss['border'][-1]:.3e}"
        )
        print(
            f"Train : loss: {train_loss['total'][-1]:.3e}, data: {train_loss['data'][-1]:.3e}, pde: {train_loss['pde'][-1]:.3e}, border: {train_loss['border'][-1]:.3e}",
            file=f,
        )
        print(
            f"Test  : loss: {test_loss['total'][-1]:.3e}, data: {test_loss['data'][-1]:.3e}, pde: {test_loss['pde'][-1]:.3e}, border: {test_loss['border'][-1]:.3e}"
        )
        print(
            f"Test  : loss: {test_loss['total'][-1]:.3e}, data: {test_loss['data'][-1]:.3e}, pde: {test_loss['pde'][-1]:.3e}, border: {test_loss['border'][-1]:.3e}",
            file=f,
        )
        print(
            f"Weights  : data: {weight_data}, pde: {weight_pde}, border: {weight_border}"
        )
        print(
            f"Weights  : data: {weight_data}, pde: {weight_pde}, border: {weight_border}",
            file=f,
        )

        print(f"time: {time.time()-time_start:.0f}s")
        print(f"time: {time.time()-time_start:.0f}s", file=f)

        if (epoch + 1) % save_rate == 0:
            dossier_midle = Path(
                folder_result + f"/epoch{len(train_loss['total'])}")
            dossier_midle.mkdir(parents=True, exist_ok=True)
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "scheduler_state_dict": scheduler.state_dict(),
                    "weights": {
                        "weight_data": weight_data,
                        "weight_pde": weight_pde,
                        "weight_border": weight_border
                    }
                },
                folder_result
                + f"/epoch{len(train_loss['total'])}"
                + "/model_weights.pth",
            )

            write_csv(
                train_loss,
                folder_result + f"/epoch{len(train_loss['total'])}",
                file_name="/train_loss.csv",
            )
            write_csv(
                test_loss,
                folder_result + f"/epoch{len(train_loss['total'])}",
                file_name="/test_loss.csv",
            )
