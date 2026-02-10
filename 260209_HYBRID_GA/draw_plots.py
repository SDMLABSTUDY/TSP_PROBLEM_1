import pandas as pd
import matplotlib.pyplot as plt
import os
import glob


def draw_representative_plot(result_folder_path, best_seed):

    # 1. CSV 파일 읽기
    csv_file = os.path.join(result_path, "logs", f"trace_seed_{best_seed}.csv")
    if not os.path.exists(csv_file):
        print(f"Error: 파일을 찾을 수 없습니다 -> {csv_file}")
        return

    df = pd.read_csv(csv_file)

    # 2. 그래프 그리기 (2개를 나란히 그림)
    fig, ax1 = plt.subplots(figsize=(10, 6))

    # X축: Generation
    line1 = ax1.plot(df['Generation'], df['Best_Fitness'], 'r-', label='Best Fitness')
    line2 = ax1.plot(df['Generation'], df['Avg_Fitness'], 'b--', label='Avg Fitness', alpha=0.5)

    ax1.set_xlabel('Generation')
    ax1.set_ylabel('Fitness (Distance)')
    ax1.set_title(f'[Representative Run] Seed {best_seed}')
    ax1.grid(True, linestyle='--', alpha=0.5)

    # 범례 합치기
    lines = line1 + line2
    labels = [l.get_label() for l in lines]
    ax1.legend(lines, labels, loc='upper right')

    # 저장
    save_path = os.path.join(result_path, "plots", "representative_plot_gen.png")
    plt.savefig(save_path)
    print(f"Saved: {save_path}")

    # 3. 추가 요구사항: Time vs Best 그래프
    plt.figure(figsize=(10, 6))
    plt.plot(df['Time'], df['Best_Fitness'], 'g-', label='Best Fitness over Time')
    plt.xlabel('Time (seconds)')
    plt.ylabel('Best Fitness')
    plt.title(f'[Time vs Best] Seed {best_seed}')
    plt.grid(True)
    plt.legend()

    save_path_time = os.path.join(result_path, "plots", "representative_plot_time.png")
    plt.savefig(save_path_time)
    print(f"Saved: {save_path_time}")


# ==========================================
# 사용법
# ==========================================
if __name__ == "__main__":
    # 예시: 방금 돌린 실험 폴더 경로를 여기에 복사해넣으세요
    # 예: "results/run_GAwithgreedy/cycle101_experiment_20251024_120000"
    result_path = "results/run_GAwithgreedy/YOUR_FOLDER_NAME_HERE"

    # 가장 결과가 좋았던 시드 번호 (Summary 보고 결정)
    best_seed_num = 0

    draw_representative_plot(result_path, best_seed_num)