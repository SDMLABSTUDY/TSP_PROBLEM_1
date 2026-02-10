import os
import math
import random
import csv
import datetime
import matplotlib.pyplot as plt


class TSP_Base:
    def __init__(self, file_path, run_name="default_run", seed=0):
        self.run_name = run_name
        self.rng = random.Random(seed)
        self.cities = []
        self.n_cities = 0
        self.limit_time = 0
        self.file_path = file_path

        # 데이터 로드
        self.load_data(file_path)
        self.dist_matrix = self.precompute_distances()

        # 결과 저장 경로
        self.result_path = self._prepare_directory()

    def _prepare_directory(self):
        # results / 알고리즘이름 / 파일명_experiment_날짜_시간
        base_dir = "results"
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

        filename = os.path.basename(self.file_path)
        dataset_name = os.path.splitext(filename)[0]

        folder_name = f"{dataset_name}_experiment_{timestamp}"
        exp_dir = os.path.join(base_dir, self.run_name, folder_name)

        if not os.path.exists(os.path.join(exp_dir, "logs")):
            os.makedirs(os.path.join(exp_dir, "logs"))
        if not os.path.exists(os.path.join(exp_dir, "plots")):
            os.makedirs(os.path.join(exp_dir, "plots"))

        return exp_dir

    def load_data(self, file_path):
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")

        with open(file_path, "r", encoding="utf-8") as f:
            lines = [ln.strip() for ln in f.read().splitlines() if ln.strip()]

        self.n_cities = int(float(lines[0]))
        self.limit_time = float(lines[-1])
        coord_lines = lines[1:1 + self.n_cities]
        self.cities = []
        for ln in coord_lines:
            parts = ln.split()
            self.cities.append((float(parts[0]), float(parts[1])))

    def precompute_distances(self):
        matrix = [[0.0] * self.n_cities for _ in range(self.n_cities)]
        for i in range(self.n_cities):
            for j in range(i + 1, self.n_cities):
                d = math.sqrt((self.cities[i][0] - self.cities[j][0]) ** 2 +
                              (self.cities[i][1] - self.cities[j][1]) ** 2)
                matrix[i][j] = matrix[j][i] = d
        return matrix

    def get_fitness(self, tour):
        distance = 0
        for i in range(self.n_cities):
            distance += self.dist_matrix[tour[i]][tour[(i + 1) % self.n_cities]]
        return distance

    def greedy_solution(self, start_node=0):
        n = self.n_cities
        dm = self.dist_matrix

        visited = [False] * n
        tour = [start_node]
        visited[start_node] = True
        cur = start_node

        for _ in range(n - 1):
            nxt = -1
            min_dist = float('inf')

            for j in range(n):
                if not visited[j]:
                    if dm[cur][j] < min_dist:
                        min_dist = dm[cur][j]
                        nxt = j
            tour.append(nxt)
            visited[nxt] = True
            cur = nxt

        return tour, self.get_fitness(tour)

    def two_opt(self, tour):
        # 2-OPT Local Search
        best_tour = tour[:]
        improved = True
        n = self.n_cities
        dm = self.dist_matrix

        while improved:
            improved = False
            for i in range(n - 1):
                for j in range(i + 2, n):
                    if j == n - 1 and i == 0: continue

                    a, b = best_tour[i], best_tour[i + 1]
                    c, d = best_tour[j], best_tour[(j + 1) % n]

                    # 거리 비교 (Delta Cost)
                    if dm[a][c] + dm[b][d] < dm[a][b] + dm[c][d]:
                        best_tour[i + 1:j + 1] = best_tour[i + 1:j + 1][::-1]
                        improved = True

        return best_tour, self.get_fitness(best_tour)

    # GPT : 저장 및 시각화 관련
    def save_seed_log(self, seed, trace_data):
        # 1. CSV 저장
        csv_path = os.path.join(self.result_path, "logs", f"trace_seed_{seed}.csv")
        with open(csv_path, "w", newline='') as f:
            writer = csv.writer(f)
            writer.writerow(["Generation", "Time", "Best_Fitness", "Avg_Fitness"])
            writer.writerows(trace_data)

        # 2. Log 파일 저장
        log_path = os.path.join(self.result_path, "logs", f"trace_seed_{seed}.log")
        with open(log_path, "w", encoding="utf-8") as f:
            f.write(f"==========================================\n")
            f.write(f"[Experiment Log] Run: {self.run_name}\n")
            f.write(f"Seed: {seed}\n")
            f.write(f"Date: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"==========================================\n")

            f.write(f"{'Gen':<6} | {'Time(s)':<10} | {'Best':<12} | {'Avg':<12}\n")
            f.write("-" * 48 + "\n")

            for row in trace_data:
                # row = (gen, time, best, avg)
                f.write(f"{row[0]:<6} | {row[1]:<10.4f} | {row[2]:<12.4f} | {row[3]:<12.4f}\n")

            f.write("=" * 48 + "\n")
            f.write(f"End of Log\n")

    def plot_solution(self, tour, title, filename):
        plt.figure(figsize=(6, 6))
        path_x = [self.cities[i][0] for i in tour] + [self.cities[tour[0]][0]]
        path_y = [self.cities[i][1] for i in tour] + [self.cities[tour[0]][1]]

        plt.plot(path_x, path_y, 'b-', marker='o', markersize=4, linewidth=1, label='Path')
        plt.plot(path_x[0], path_y[0], 'ro', markersize=8, label='Start')

        plt.title(title)
        plt.xlabel('X')
        plt.ylabel('Y')
        plt.grid(True)
        plt.legend()

        save_path = os.path.join(self.result_path, "plots", filename)
        plt.savefig(save_path)
        plt.close()

    def save_summary(self, header, data):
        path = os.path.join(self.result_path, "summary_results.csv")
        with open(path, "w", newline='') as f:
            writer = csv.writer(f)
            writer.writerow(header)
            writer.writerows(data)
        print(f"\n[Done] Results saved to: {self.result_path}")


# =========================================================
# GA Engine 클래스
# =========================================================
class GA_Engine(TSP_Base):
    def __init__(self, file_path, params, run_name, seed):
        super().__init__(file_path, run_name, seed)
        self.params = params

    def selection(self, population):
        t = self.params['TOURNAMENT_T']
        pop_len = len(population)

        def tournament():
            idx1 = self.rng.randrange(pop_len)
            idx2 = self.rng.randrange(pop_len)

            ch1, fit1 = population[idx1]
            ch2, fit2 = population[idx2]

            # 더 좋은 개체를 ch1으로 설정
            if fit2 < fit1:
                ch1, ch2 = ch2, ch1

            # t의 확률로 ch1 선택
            if self.rng.random() < t:
                return ch1
            else:
                return ch2

        return tournament(), tournament()

    def crossover(self, mom, dad):
        n = self.n_cities
        l, r = sorted(self.rng.sample(range(n), 2))
        child = [-1] * n
        child[l:r + 1] = mom[l:r + 1]

        used = set(child[l:r + 1])

        idx = (r + 1) % n
        for city in dad:
            if city in used:
                continue
            while child[idx] != -1:
                idx = (idx + 1) % n

            child[idx] = city

        return child

    def mutation(self, tour):
        n = self.n_cities
        i, j = sorted(self.rng.sample(range(n), 2))
        tour[i:j + 1] = reversed(tour[i:j + 1])

        return tour

    def replacement(self, population, offsprings):
        merged = population + offsprings
        merged.sort(key=lambda x: x[1])
        return merged[:self.params['POP_SIZE']]