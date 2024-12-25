import logging, torch, numpy as np
from tqdm import tqdm
from time import time
from . loss_fun import FocalLoss

from . import utils as U
from .initializer import Initializer


class Processor(Initializer):

    def train(self, epoch):
        self.model.train()
        start_train_time = time()
        num_top1, num_sample = 0, 0
        counts_pre_0, counts_pre_1, counts_truth_0, counts_truth_1 = 0,0,0,0
        train_iter = self.train_loader if self.no_progress_bar else tqdm(self.train_loader, dynamic_ncols=True)
        for num, (x, y, _) in enumerate(train_iter):
            self.optimizer.zero_grad()

            # Using GPU
            x = x.float().to(self.device)
            y = y.long().to(self.device)

            # Calculating Output
            out, _ = self.model(x)

            # Updating Weights
            loss = self.loss_func(out, y)
            # criterion = FocalLoss().to(self.device)
            # loss2 = criterion(out,y)
            # loss = 0.5*loss1 + 0.5*loss2
            loss.backward()
            self.optimizer.step()
            self.scheduler.step()
            self.global_step += 1

            # Calculating Recognition Accuracies
            num_sample += x.size(0)
            reco_top1 = out.max(dim = 1)[1]
            # print(reco_top1.eq(y))
            num_top1 += reco_top1.eq(y).sum().item()
            train_acc = num_top1 / num_sample
            # print('train_acc:',train_acc)

            # 统计预测出来的结果和真实标签不同的数量
            diff_mask  = reco_top1 != y
            different_values_in_pred = reco_top1[diff_mask]
            different_values_in_truth = y[diff_mask]
            # print("pred中不同位置的元素:", different_values_in_pred)
            # print("truth中对应位置的元素:", different_values_in_truth)

            # 计算pred中0和1的数量
            counts_pre_0 += (different_values_in_pred == 0).sum().item()
            counts_pre_1 += (different_values_in_pred == 1).sum().item()
            # 计算truth中0和1的数量
            counts_truth_0 += (different_values_in_truth == 0).sum().item() ## 正样本，有动作
            counts_truth_1 += (different_values_in_truth == 1).sum().item() ## 负样本， 无动作
            # 输出每个数组中0和1的数量
            # print("pred中0的数量:", counts_pre_0)
            # print("pred中1的数量:", counts_pre_1)
            # print("truth中0的数量:", counts_truth_0)
            # print("truth中1的数量:", counts_truth_1)
            #
            #
            # print("正样本预测成负样本的个数——>举手预测成非举手", counts_truth_1, counts_pre_0)
            # print("负样本预测成正样本的数量——>非举手预测成举手", counts_truth_0, counts_pre_1)


            # Showing Progress
            lr = self.optimizer.param_groups[0]['lr']
            # print(lr)
            if self.scalar_writer:
                self.scalar_writer.add_scalar('learning_rate', lr, self.global_step)
                self.scalar_writer.add_scalar('train_loss', loss.item(), self.global_step)
            if self.no_progress_bar:
                logging.info('Epoch: {}/{}, Batch: {}/{}, Loss: {:.4f}, LR: {:.4f}'.format(
                    epoch+1, self.max_epoch, num+1, len(self.train_loader), loss.item(), lr
                ))
            else:
                train_iter.set_description('Loss: {:.4f}, LR: {:.4f}, train_acc: {:.4f}， positive_negative: {:d}， negative_positive: {:d}'\
                                           .format(loss.item(), lr, train_acc, counts_truth_0, counts_truth_1))
                # print(lr)

        # Showing Train Results
        train_acc = num_top1 / num_sample
        if self.scalar_writer:
            self.scalar_writer.add_scalar('train_acc', train_acc, self.global_step)
        logging.info('Epoch: {}/{}, Training accuracy: {:d}/{:d}({:.2%}), Training time: {:.2f}s'.format(
            epoch+1, self.max_epoch, num_top1, num_sample, train_acc, time()-start_train_time
        ))
        logging.info('')


    def eval(self):
        self.model.eval()
        start_eval_time = time()
        with torch.no_grad():
            num_top1, num_top2 = 0, 0
            counts_pre_0, counts_pre_1, counts_truth_0, counts_truth_1 = 0, 0, 0, 0
            num_sample, eval_loss = 0, []
            cm = np.zeros((self.num_class, self.num_class))

            # !!!!验证结果时输出分布，训练时注释掉
            # accLog = []
            # CacheNumpy = []
            # softmax = torch.nn.Softmax(dim=1)

            eval_iter = self.eval_loader if self.no_progress_bar else tqdm(self.eval_loader, dynamic_ncols=True)
            for num, (x, y, _) in enumerate(eval_iter):

                # !!!!验证结果时输出分布，训练时注释掉
                # CacheNumpy.append(x.float().numpy())

                # Using GPU
                x = x.float().to(self.device)
                y = y.long().to(self.device)
                # Calculating Output
                out, _ = self.model(x)

                # !!!!验证结果时输出分布，训练时注释掉
                # outSoftmax = softmax(out)
                # accLog.append(outSoftmax)

                # Getting Loss
                loss = self.loss_func(out, y)
                # criterion = FocalLoss().to(self.device)
                # loss2 = criterion(out, y)
                # loss = 0.5*loss1 + 0.5*loss2
                eval_loss.append(loss.item())
                # Calculating Recognition Accuracies
                num_sample += x.size(0)
                reco_top1 = out.max(1)[1]
                num_top1 += reco_top1.eq(y).sum().item()
                reco_top2 = torch.topk(out,2)[1]
                num_top2 += sum([y[n] in reco_top2[n,:] for n in range(x.size(0))])

                # 统计预测出来的结果和真实标签不同的数量
                diff_mask = reco_top1 != y
                different_values_in_pred = reco_top1[diff_mask]
                different_values_in_truth = y[diff_mask]
                # print("pred中不同位置的元素:", different_values_in_pred)
                # print("truth中对应位置的元素:", different_values_in_truth)

                # 计算pred中0和1的数量
                counts_pre_0 += (different_values_in_pred == 0).sum().item()
                counts_pre_1 += (different_values_in_pred == 1).sum().item()
                # 计算truth中0和1的数量
                counts_truth_0 += (different_values_in_truth == 0).sum().item()
                counts_truth_1 += (different_values_in_truth == 1).sum().item()
                # 输出每个数组中0和1的数量
                # print("pred中0的数量:", counts_pre_0)
                # print("pred中1的数量:", counts_pre_1)
                # print("truth中0的数量:", counts_truth_0)
                # print("truth中1的数量:", counts_truth_1)
                #
                # print("正样本预测成负样本的个数——>举手预测成非举手", counts_truth_1, counts_pre_0)
                # print("负样本预测成正样本的数量——>非举手预测成举手", counts_truth_0, counts_pre_1)

                # Calculating Confusion Matrix
                for i in range(x.size(0)):
                    cm[y[i], reco_top1[i]] += 1
                # Showing Progress
                if self.no_progress_bar and self.args.evaluate:
                    logging.info('Batch: {}/{}'.format(num+1, len(self.eval_loader)))
                else:
                    eval_iter.set_description(
                        ' positive_negative: {:d}， negative_positive: {:d}' \
                        .format(counts_truth_0, counts_truth_1))

        # Showing Evaluating Results
        acc_top1 = num_top1 / num_sample
        acc_top2 = num_top2 / num_sample
        eval_loss = sum(eval_loss) / len(eval_loss)
        eval_time = time() - start_eval_time
        eval_speed = len(self.eval_loader) * self.eval_batch_size / eval_time / len(self.args.gpus)
        logging.info('Top-1 accuracy: {:d}/{:d}({:.2%}), Top-2 accuracy: {:d}/{:d}({:.2%}), Mean loss:{:.4f}'.format(
            num_top1, num_sample, acc_top1, num_top2, num_sample, acc_top2, eval_loss
        ))
        logging.info('Evaluating time: {:.2f}s, Speed: {:.2f} sequnces/(second*GPU)'.format(
            eval_time, eval_speed
        ))
        logging.info('')
        if self.scalar_writer:
            self.scalar_writer.add_scalar('eval_acc', acc_top1, self.global_step)
            self.scalar_writer.add_scalar('eval_loss', eval_loss, self.global_step)

        # !!!!验证结果时输出分布，训练时注释掉
        # acc = torch.cat(accLog, dim=0)[:, 0].tolist()
        # from collections import Counter
        # ret = Counter(acc)
        # f = open('accRet.csv', 'w')
        # for i in ret.keys():
        #     f.writelines(f'{i},{ret[i]}\n')
        # f.close()
        #CacheOut = np.concatenate(CacheNumpy, axis=0)
        #np.save('cache.npy',CacheOut)

        torch.cuda.empty_cache()
        return acc_top1, acc_top2, cm


    def start(self):
        start_time = time()
        if self.args.evaluate:
            if self.args.debug:
                logging.warning('Warning: Using debug setting now!')
                logging.info('')

            # Loading Evaluating Model
            logging.info('Loading evaluating model ...')
            checkpoint = U.load_checkpoint(self.args.work_dir, self.model_name)
            if checkpoint:
                self.model.module.load_state_dict(checkpoint['model'])
            logging.info('Successful!')
            logging.info('')

            # Evaluating
            logging.info('Starting evaluating ...')
            self.eval()
            logging.info('Finish evaluating!')

        else:
            # Resuming
            start_epoch = 0
            best_state = {'acc_top1':0, 'acc_top2':0, 'cm':0}
            if self.args.resume:
                logging.info('Loading checkpoint ...')
                checkpoint = U.load_checkpoint(self.args.work_dir)
                self.model.module.load_state_dict(checkpoint['model'])
                self.optimizer.load_state_dict(checkpoint['optimizer'])
                # 更新一下学习率，学习率更新为0.0001 lr = self.optimizer.param_groups[0]['lr']
                # self.optimizer.param_groups[0]['lr'] = 0.0001

                self.scheduler.load_state_dict(checkpoint['scheduler'])
                start_epoch = checkpoint['epoch']
                checkpoint['best_state'] = {'acc_top1':0.5, 'acc_top2':0, 'cm':0} # 接续训练，将最优的准确率置为0
                print(checkpoint['best_state'])
                best_state.update(checkpoint['best_state'])
                self.global_step = start_epoch * len(self.train_loader)
                logging.info('Start epoch: {}'.format(start_epoch+1))
                logging.info('Best accuracy: {:.2%}'.format(best_state['acc_top1']))
                logging.info('Successful!')
                logging.info('')

            # Training
            logging.info('Starting training ...')
            for epoch in range(start_epoch, self.max_epoch):

                # Training
                self.train(epoch)

                # Evaluating
                is_best = False
                # if (epoch+1) % self.eval_interval(epoch) == 0:
                if (epoch + 1) % 1 == 0:
                    logging.info('Evaluating for epoch {}/{} ...'.format(epoch+1, self.max_epoch))
                    acc_top1, acc_top2, cm = self.eval()
                    if acc_top1 > best_state['acc_top1']:
                        is_best = True
                        best_state.update({'acc_top1':acc_top1, 'acc_top2':acc_top2, 'cm':cm})

                # Saving Model
                logging.info('Saving model for epoch {}/{} ...'.format(epoch+1, self.max_epoch))
                U.save_checkpoint(
                    self.model.module.state_dict(), self.optimizer.state_dict(), self.scheduler.state_dict(),
                    epoch+1, best_state, is_best, self.args.work_dir, self.save_dir, self.model_name
                )
                logging.info('Best top-1 accuracy: {:.2%}, Total time: {}'.format(
                    best_state['acc_top1'], U.get_time(time()-start_time)
                ))
                logging.info('')
            logging.info('Finish training!')
            logging.info('')

    def extract(self):
        logging.info('Starting extracting ...')
        if self.args.debug:
            logging.warning('Warning: Using debug setting now!')
            logging.info('')

        # Loading Model
        logging.info('Loading evaluating model ...')
        checkpoint = U.load_checkpoint(self.args.work_dir, self.model_name)
        if checkpoint:
            self.cm = checkpoint['best_state']['cm']
            self.model.module.load_state_dict(checkpoint['model'])
        logging.info('Successful!')
        logging.info('')

        # Loading Data
        x, y, names = iter(self.eval_loader).next()
        location = self.location_loader.load(names) if self.location_loader else []

        # Calculating Output
        self.model.eval()
        out, feature = self.model(x.float().to(self.device))

        # Processing Data
        data, label = x.numpy(), y.numpy()
        out = torch.nn.functional.softmax(out, dim=1).detach().cpu().numpy()
        weight = self.model.module.classifier.fc.weight.squeeze().detach().cpu().numpy()
        feature = feature.detach().cpu().numpy()

        # Saving Data
        if not self.args.debug:
            U.create_folder('./visualization')
            np.savez('./visualization/extraction_{}.npz'.format(self.args.config),
                data=data, label=label, name=names, out=out, cm=self.cm,
                feature=feature, weight=weight, location=location
            )
        logging.info('Finish extracting!')
        logging.info('')
