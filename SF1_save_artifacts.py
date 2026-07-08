
import pickle
from pathlib import Path
import torch


def save_training_artifacts(model, optimizer=None, run_config=None, metrics=None,
                            model_pt_path='output/model_state.pt',
                            checkpoint_pt_path='output/checkpoint.pt',
                            run_pickle_path='output/run_info.pkl'):
    Path('output').mkdir(exist_ok=True)

    torch.save(model.state_dict(), model_pt_path)

    checkpoint = {
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict() if optimizer is not None else None,
        'run_config': run_config if run_config is not None else {},
        'metrics': metrics if metrics is not None else {},
    }
    torch.save(checkpoint, checkpoint_pt_path)

    run_info = {
        'run_config': run_config if run_config is not None else {},
        'metrics': metrics if metrics is not None else {},
        'model_pt_path': str(model_pt_path),
        'checkpoint_pt_path': str(checkpoint_pt_path),
    }
    with open(run_pickle_path, 'wb') as f:
        pickle.dump(run_info, f)

    return {
        'model_pt_path': str(model_pt_path),
        'checkpoint_pt_path': str(checkpoint_pt_path),
        'run_pickle_path': str(run_pickle_path),
    }


# Example usage:
# artifacts = save_training_artifacts(
#     model=model,
#     optimizer=optimizer,
#     run_config={
#         'hidden_channels': 64,
#         'num_layers': 3,
#         'activation': 'relu',
#         'learning_rate': 1e-3,
#         'batch_size': 8,
#         'weight_decay': 1e-4,
#         'target': 'von_mises'
#     },
#     metrics={
#         'train_loss_last': 0.0012,
#         'val_mae': 0.034,
#         'test_rmse': 0.052
#     },
#     model_pt_path='output/graphsage_model_state.pt',
#     checkpoint_pt_path='output/graphsage_checkpoint.pt',
#     run_pickle_path='output/graphsage_run_info.pkl'
# )
# print(artifacts)
