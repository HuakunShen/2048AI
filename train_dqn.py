import torch
from pathlib import Path
from src.game.controller.game import Game
from src.agent.agentImpl import DQNPlayer
from src.game.model.staticboardImpl import TorchStaticBoard
import time


def get_device():
    """Get the best available device (MPS for Apple Silicon, CUDA for NVIDIA, or CPU)"""
    if torch.backends.mps.is_available():
        return torch.device("mps")
    elif torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def main():
    # Set up device
    device = get_device()
    print(f"Using device: {device}")
    
    # Create directories for models and logs
    Path("models").mkdir(parents=True, exist_ok=True)
    timestamp = int(time.time())
    model_name = f"dqn_2048_{timestamp}"
    checkpoint_dir = Path(f"models/{model_name}")
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    # Initialize game with TorchStaticBoard for tensor operations
    game = Game(static_board=TorchStaticBoard)
    
    # Create DQN agent with improved hyperparameters
    agent = DQNPlayer(
        game=game,
        quiet=False,
        ui=True,
        memory_size=100000,  # Increased memory size
        batch_size=256,      # Larger batch size
        gamma=0.95,          # Slightly reduced discount factor
        epsilon=1.0,
        epsilon_min=0.01,
        epsilon_decay=0.995,
        learning_rate=0.0005,  # Reduced learning rate
        device=device
    )
    
    # Training configuration
    num_episodes = 50
    save_frequency = 100  # Save model every 100 episodes
    
    print("Starting training...")
    try:
        for episode in range(0, num_episodes, save_frequency):
            # Train for save_frequency episodes
            agent.train(episodes=save_frequency, target_update=10)
            
            # Save checkpoint
            checkpoint_path = checkpoint_dir / f"checkpoint_episode_{episode + save_frequency}.pth"
            agent.save_model(str(checkpoint_path))
            print(f"Saved checkpoint at episode {episode + save_frequency}")
            
        # Save final model
        agent.save_model(str(checkpoint_dir / "final_model.pth"))
        
    except KeyboardInterrupt:
        print("\nTraining interrupted. Saving current model...")
        agent.save_model(str(checkpoint_dir / "interrupted_model.pth"))
    
    print("\nTraining completed. Testing final model...")
    game.restart()
    score, max_val, runtime = agent.run()
    print(f"Test Results - Score: {score}, Max Value: {max_val}, Runtime: {runtime:.2f}s")
    
    # Save final test results
    with open(checkpoint_dir / "test_results.txt", "w") as f:
        f.write(f"Score: {score}\nMax Value: {max_val}\nRuntime: {runtime:.2f}s")


if __name__ == "__main__":
    main()
