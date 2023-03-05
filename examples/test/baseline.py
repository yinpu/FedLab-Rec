import sys
sys.path.append("../..")
from fedlab_rec.data.amazon import AmazonDataSet
# from fedlab_rec.data.utils import df_to_dataloader
# from fedlab_rec.model.ncf import NCF
from fedlab_rec.trainer.ranker_trainer import CTRTrainer
from fedlab_rec.utils.metric import topk_metrics



if __name__=="__main__":
    dataset = AmazonDataSet(root_dir='../../data/',
                            dataset_name='Prime_Pantry',
                            batch_size=1024)
    train_df, test_df = dataset.train_df, dataset.test_df
    print(train_df.head(5))
    print(test_df.head(5))
    train_loader, test_loader = df_to_dataloader(train_df, 1024), df_to_dataloader(test_df, 1024)
    model = NCF(dataset.users_num, dataset.items_num)
    trainer = CTRTrainer(evaluate_fn=topk_metrics,
                        n_epoch=400,
                        device="cpu")
    trainer.setup(model)
    # trainer.fit(train_loader)
    # print(trainer.evaluate(test_loader))
    for epoch_i in range(400):
        print('epoch:', epoch_i)
        trainer.train_one_epoch(train_loader)
        print(trainer.evaluate(test_loader))
    
    
    
    