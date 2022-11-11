import itertools
import discriminator
import generator
import utils
import torch
import torch.nn as nn
import torch.nn.functional as f
from torch.utils.data import DataLoader
import config


def run_train():
    train_data=DataLoader(
    utils.Data_load("../input/az-handwritten-alphabets-in-csv-format/A_Z Handwritten Data.csv",
    mode='train',
    trans=utils.transformations),
    batch_size=config.train_batch_size,
    shuffle=True)

    test_data=DataLoader(
    utils.Data_load("../input/az-handwritten-alphabets-in-csv-format/A_Z Handwritten Data.csv",
    mode='test',
    trans=utils.transformations),
    batch_size=config.test_batch_size,
    shuffle=False)
    
    
    # criterions
    criterion_gan = nn.MSELoss()
    criterion_cycle = nn.L1Loss()
    criterion_identity = nn.L1Loss()

    #changing to device
    criterion_gan.to(config.device)
    criterion_cycle.to(config.device)
    criterion_identity.to(config.device)

    #model
    G_ab = generator.Generator(1)
    G_ba = generator.Generator(1)
    D_a = discriminator.Discriminator(1)
    D_b = discriminator.Discriminator(1)

    #changing them to device
    G_ab = G_ab.to(config.device)
    G_ba = G_ba.to(config.device)
    D_a = D_a.to(config.device)
    D_b = D_b.to(config.device)


    #optimizer
    lr = config.lr
    b1 = config.b1
    b2 = config.b2

    optim_G = torch.optim.Adam(itertools.chain(
        G_ab.parameters(), G_ba.parameters()), lr=lr, betas=(b1, b2))
    optim_Da = torch.optim.Adam(D_a.parameters(), lr=lr, betas=(b1, b2))
    optim_Db = torch.optim.Adam(D_b.parameters(), lr=lr, betas=(b1, b2))


    #lr scheduler
    epochs = config.epochs
    decay_epoch = config.decay_epoch


    def lambda_func(epoch): return 1 - max(0, epoch -
                                        decay_epoch)/(epochs-decay_epoch)


    lr_scheduler_G = torch.optim.lr_scheduler.LambdaLR(
        optim_G, lr_lambda=lambda_func)
    lr_scheduler_D_A = torch.optim.lr_scheduler.LambdaLR(
        optim_Da, lr_lambda=lambda_func)
    lr_scheduler_D_B = torch.optim.lr_scheduler.LambdaLR(
        optim_Db, lr_lambda=lambda_func)


    for epoch in range(epochs):
        for i, (a, b) in enumerate(train_data):
            a, b = a.to(config.device), b.to(config.device)
            shape = [int(a.size(0)), 1, int(a.size(2)/4), int(a.size(3)/4)]
            real_label = torch.ones(shape, device=config.device)
            fake_label = torch.zeros(shape, device=config.device)

            G_ab.train()
            G_ba.train()

            fake_b = G_ab(a)
            fake_a = G_ba(b)

            #loss beetween in _im and out_im
            id_1 = criterion_identity(fake_b, a)
            id_2 = criterion_identity(fake_a, b)
            main_id = (id_1+id_2)/2

            #missguiding
            gan_1 = criterion_gan(D_a(fake_a), real_label)
            gan_2 = criterion_gan(D_b(fake_b), real_label)
            main_gan = (gan_1+gan_2)/2

            cy_1 = criterion_cycle(G_ba(fake_b), a)
            cy_2 = criterion_cycle(G_ab(fake_a), b)
            main_cy = (cy_1+cy_2)/2

            gan_loss = 5*main_id + main_gan + 10*main_cy
            optim_G.zero_grad()
            gan_loss.backward()
            optim_G.step()

            #trainnig_D_a
            optim_Da.zero_grad()
            da_1 = criterion_gan(D_a(fake_a.detach()), fake_label)
            da_2 = criterion_gan(D_a(a), real_label)
            loss_da = (da_1+da_2)/2
            loss_da.backward()
            optim_Da.step()
            #training D_b
            optim_Db.zero_grad()
            db_1 = criterion_gan(D_b(fake_b.detach()), fake_label)
            db_2 = criterion_gan(D_b(b), real_label)
            loss_db = (db_1+db_2)/2
            loss_db.backward()
            optim_Db.step()

        lr_scheduler_G.step()
        lr_scheduler_D_A.step()
        lr_scheduler_D_B.step()

        with torch.no_grad():

            if (epoch+1) % 10 == 0:

                loss_d = (loss_da+loss_db)/2
                print(f'''
                epoch = {epoch+1}/{epochs}
                gan_loss = {gan_loss.item()}
                subsidiaries:
                id={main_id.item()} , gan={main_gan.item()} , cy={main_cy.item()}
    
                total_d_loss = {loss_d.item()}
                subsidiaries:
                da={loss_da.item()} , db={loss_db.item()}
                ''')

                a, b = next(iter(train_data))
                utils.plot(a, b)
