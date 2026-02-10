import pandas as pd
import matplotlib.pyplot as plt
import os
import glob

# =========================================================
# 사용 설정 (여기에 분석하고 싶은 폴더 경로를 넣으세요)
# =========================================================
# 예: "results/run_GAwithoutgreedy/cycle100_experiment_20250211_120000"
TARGET_DIR = r"results/run_GAwithgreedy/cycle100_experiment_YYYYMMDD_HHMMSS"  # <-- 여기에 경로 복사!
REPRESENTATIVE_SEED = 0  # 결과가 가장 좋거나 평균적인 '대표' 시드 번호


def plot_representative_run(target_dir, seed):
    """
    [과제 6.2 필수] 대표 Run 1개에 대한 상세 분석 그래프
    1. Generation vs (Best & Avg)
    2. Time vs Best
    """
    csv_path = os.path.join(target_dir, "logs", f"trace_seed_{seed}.csv")

    if not os.path.exists(csv_path):
        print(f"❌ Error: 해당 시드의 로그 파일이 없습니다.\n경로: {csv_path}")
        return

    # CSV 읽기
    df = pd.read_csv(csv_path)

    # -------------------------------------------------------
    # 1. Generation 기준 수렴 그래프 (Best vs Avg)
    # -------------------------------------------------------
    plt.figure(figsize=(10, 6))

    plt.plot(df['Generation'], df['Avg_Fitness'], 'b--', label='Population Average', alpha=0.5, linewidth=1.5)
    plt.plot(df['Generation'], df['Best_Fitness'], 'r-', label='Population Best', linewidth=2)

    plt.title(f"[Representative Run] Convergence by Generation (Seed {seed})", fontsize=14)
    plt.xlabel("Generation", fontsize=12)
    plt.ylabel("Total Distance (Fitness)", fontsize=12)
    plt.legend(fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.6)

    save_path_gen = os.path.join(target_dir, "plots", f"report_6-2_gen_seed{seed}.png")
    plt.savefig(save_path_gen, dpi=150)
    print(f"✅ Saved: {save_path_gen}")
    plt.close()

    # -------------------------------------------------------
    # 2. Time 기준 수렴 그래프 (Time vs Best) - 설득력 강화용
    # -------------------------------------------------------
    plt.figure(figsize=(10, 6))

    plt.plot(df['Time'], df['Best_Fitness'], 'g-', label='Best Fitness over Time', linewidth=2)

    plt.title(f"[Analysis] Improvement over Time (Seed {seed})", fontsize=14)
    plt.xlabel("Time (seconds)", fontsize=12)
    plt.ylabel("Best Fitness", fontsize=12)
    plt.legend(fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.6)

    save_path_time = os.path.join(target_dir, "plots", f"report_6-2_time_seed{seed}.png")
    plt.savefig(save_path_time, dpi=150)
    print(f"✅ Saved: {save_path_time}")
    plt.close()


def plot_all_seeds_comparison(target_dir):
    """
    [추가 분석] 모든 시드의 Best 수렴 곡선을 한 번에 비교
    - 어떤 시드는 빠르고, 어떤 시드는 느린지 분포 확인용
    """
    log_pattern = os.path.join(target_dir, "logs", "trace_seed_*.csv")
    files = glob.glob(log_pattern)

    if not files:
        print("❌ 로그 파일들을 찾을 수 없습니다.")
        return

    plt.figure(figsize=(12, 7))

    for f in files:
        # 파일명에서 시드 번호 추출
        seed_num = os.path.basename(f).split('_')[-1].replace('.csv', '')
        df = pd.read_csv(f)
        plt.plot(df['Generation'], df['Best_Fitness'], alpha=0.3, label=f"Seed {seed_num}")

    plt.title("Convergence Comparison (All Seeds)", fontsize=14)
    plt.xlabel("Generation")
    plt.ylabel("Best Fitness")
    # 범례가 너무 많으면 지저분하므로 생략하거나 조정
    # plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, alpha=0.5)

    save_path = os.path.join(target_dir, "plots", "analysis_all_seeds.png")
    plt.savefig(save_path, dpi=150)
    print(f"✅ Saved: {save_path}")
    plt.close()


if __name__ == "__main__":
    # 1. 경로 확인 (실제 존재하는지)
    if os.path.exists(TARGET_DIR):
        print(f"📂 Processing: {TARGET_DIR}")

        # 2. 대표 Run 그리기 (과제 6.2)
        plot_representative_run(TARGET_DIR, REPRESENTATIVE_SEED)

        # 3. 전체 시드 비교 그리기 (보고서 분석용)
        plot_all_seeds_comparison(TARGET_DIR)

        print("\n🎉 모든 그래프 작성이 완료되었습니다!")
    else:
        print(f"❌ 경로를 찾을 수 없습니다: {TARGET_DIR}")
        print("   코드 상단의 'TARGET_DIR' 변수를 수정해주세요.")