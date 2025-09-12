# main.py
import numpy as np
from collections import deque
import math
import matplotlib.pyplot as plt
import matplotlib.patches as patches


# --------------------------------------------------------------------------
# ЗАГЛУШКА ДЛЯ МЕТОДА TOPSIS (УЛУЧШЕННАЯ ВЕРСИЯ)
# --------------------------------------------------------------------------
def run_topsis(candidates: list, criteria_matrix: np.ndarray) -> int:
    """
    Упрощенная, но более реалистичная реализация TOPSIS.
    Возвращает индекс лучшего кандидата.
    """
    print(f"\n🚀 Запуск TOPSIS для {len(candidates)} кандидатов...")
    if not candidates:
        raise ValueError("Нет кандидатов для анализа TOPSIS.")

    # Критерий 1: Соотношение длин. Идеальное значение = 1.
    # Критерий 2: Схожесть формы. Идеальное значение = 0.

    # Преобразуем критерии так, чтобы для обоих идеальным значением был 0.
    processed_criteria = criteria_matrix.copy()
    processed_criteria[:, 0] = np.abs(processed_criteria[:, 0] - 1)

    # В реальном TOPSIS была бы нормализация и взвешивание.
    # Здесь мы просто суммируем отклонения от идеала.
    # Предполагаем, что оба критерия равнозначны.
    scores = np.sum(processed_criteria, axis=1)

    # Находим индекс кандидата с минимальной суммарной ошибкой.
    best_candidate_idx_in_list = np.argmin(scores)
    best_map_point_idx = candidates[best_candidate_idx_in_list]

    print(
        f"🏆 TOPSIS выбрал кандидата с индексом {best_map_point_idx} (общая оценка: {scores[best_candidate_idx_in_list]:.4f})")
    return best_map_point_idx


# --------------------------------------------------------------------------

class GeoMatcher:
    """
    Выполняет сопоставление объектов с фото на карте, используя
    структурный анализ маршрутов и многокритериальную оценку.
    """

    def __init__(self, map_coords: np.ndarray, photo_coords: np.ndarray):
        """
        Инициализирует матчер, преобразует координаты в точки и строит графы.
        """
        self.map_points = self._get_centers(map_coords)
        self.photo_points = self._get_centers(photo_coords)

        self.map_graph, self.map_distances = self._build_graph(self.map_points)
        self.photo_graph, self.photo_distances = self._build_graph(self.photo_points)
        print("GeoMatcher инициализирован: графы для карты и фото построены.")

    def _get_centers(self, coords: np.ndarray) -> np.ndarray:
        """Преобразует координаты [x1, y1, x2, y2] в центры [x, y]."""
        return np.array([
            (coords[:, 0] + coords[:, 2]) / 2,
            (coords[:, 1] + coords[:, 3]) / 2
        ]).T

    def _build_graph(self, points: np.ndarray):
        """Строит полностью связанный граф и кэширует расстояния между точками."""
        num_points = len(points)
        # Каждая точка соединена со всеми, кроме себя
        graph = {i: [j for j in range(num_points) if i != j] for i in range(num_points)}
        distances = {}
        for i in range(num_points):
            for j in range(i, num_points):
                dist = np.linalg.norm(points[i] - points[j])
                distances[(i, j)] = distances[(j, i)] = dist
        return graph, distances

    def _find_point_index(self, points: np.ndarray, target_coords: tuple) -> int | None:
        """Находит индекс точки по её координатам с небольшим допуском."""
        target_coords_np = np.array(target_coords)
        dists = np.linalg.norm(points - target_coords_np, axis=1)
        if np.min(dists) < 1e-6:
            return np.argmin(dists)
        return None

    def _is_in_sector(self, point_to_check: np.ndarray, cone_vertex: np.ndarray, cone_target: np.ndarray,
                      fov_degrees: float = 90.0) -> bool:
        """Проверяет, находится ли точка в секторе обзора конуса."""
        main_vector = cone_target - cone_vertex
        check_vector = point_to_check - cone_vertex

        # Чтобы избежать ошибок с нулевыми векторами
        if np.all(main_vector == 0) or np.all(check_vector == 0):
            return True

        main_angle = math.atan2(main_vector[1], main_vector[0])
        check_angle = math.atan2(check_vector[1], check_vector[0])

        angle_diff = abs(math.degrees(main_angle - check_angle))
        # Корректно обрабатываем переход через 360 градусов
        angle_diff = min(angle_diff, 360 - angle_diff)

        return angle_diff <= fov_degrees / 2

    def _find_shortest_path_bfs(self, graph, distances, start_idx, end_idx) -> tuple[list | None, float]:
        """Находит кратчайший путь по числу рёбер с помощью BFS."""
        if start_idx == end_idx:
            return [start_idx], 0.0

        queue = deque([[start_idx]])
        visited = {start_idx}

        while queue:
            path = queue.popleft()
            node = path[-1]

            if node == end_idx:
                path_length = sum(distances[(path[i], path[i + 1])] for i in range(len(path) - 1))
                return path, path_length

            for neighbor in sorted(graph[node]):
                if neighbor not in visited:
                    visited.add(neighbor)
                    new_path = list(path)
                    new_path.append(neighbor)
                    queue.append(new_path)

        # Если цикл завершился, а путь не найден
        return None, 0.0

    def _find_all_paths_dfs(self, start_idx, end_idx, path, all_paths, fov_filter):
        """Рекурсивный DFS для поиска всех простых путей в секторе."""
        current_path = path + [start_idx]

        # Проверяем, находится ли текущая точка в секторе обзора
        if not self._is_in_sector(self.map_points[start_idx], **fov_filter):
            return  # Отсекаем эту ветвь поиска

        if start_idx == end_idx:
            all_paths.append(current_path)
            return

        for neighbor in self.map_graph[start_idx]:
            if neighbor not in current_path:  # Избегаем циклов
                self._find_all_paths_dfs(neighbor, end_idx, current_path, all_paths, fov_filter)

    # В классе GeoMatcher
    def find_best_match(self, start_point_coords: tuple, target_photo_coords: tuple, known_start_photo_idx: int = None):
        start_map_idx = self._find_point_index(self.map_points, start_point_coords)

        # Используем либо известный индекс, либо ищем его
        start_photo_idx = known_start_photo_idx if known_start_photo_idx is not None else self._find_point_index(
            self.photo_points, start_point_coords)

        target_photo_idx = self._find_point_index(self.photo_points, target_photo_coords)

        if any(idx is None for idx in [start_map_idx, start_photo_idx, target_photo_idx]):
            raise ValueError("Одна из ключевых точек (старт/цель) не найдена в координатах.")

        # --- Шаг 1: Эталонный маршрут на фото (кратчайший и самый надежный) ---
        print("1. Поиск эталонного маршрута на фото...")
        ref_path, ref_length = self._find_shortest_path_bfs(self.photo_graph, self.photo_distances, target_photo_idx,
                                                            start_photo_idx)

        # ❗ ВАЖНАЯ ПРОВЕРКА, ИСПРАВЛЯЮЩАЯ ОШИБКУ
        if ref_path is None:
            raise ValueError("Не удалось построить эталонный маршрут на фото. Проверьте связность объектов.")

        print(f"   -> Найден путь: {ref_path} (длина: {ref_length:.2f}, рёбер: {len(ref_path) - 1})")

        # --- Шаг 2: Находим ВСЕ возможные маршруты на карте в секторах обзора ---
        print("\n2. Поиск всех маршрутов на карте в секторах обзора...")
        all_map_paths = []
        for i in range(len(self.map_points)):
            if i == start_map_idx: continue

            # Для каждого кандидата 'i' мы ищем пути до 'start_map_idx',
            # находясь в секторе, который "смотрит" с 'i' на 'start_map_idx'.
            fov_filter = {'cone_vertex': self.map_points[i], 'cone_target': self.map_points[start_map_idx]}
            paths_from_i = []
            self._find_all_paths_dfs(i, start_map_idx, [], paths_from_i, fov_filter)

            for p in paths_from_i:
                length = sum(self.map_distances[(p[k], p[k + 1])] for k in range(len(p) - 1))
                all_map_paths.append({'endpoint_idx': i, 'path': p, 'length': length})
        print(f"   -> Найдено {len(all_map_paths)} возможных маршрутов.")

        # Создаем словарь для отладочной информации
        debug_info = {
            "ref_path": ref_path,
            "filtered_paths": [],
            "criteria_list": np.array([]),
            "candidates": [],
            "best_match_idx": None
        }

        # --- Шаг 3: Фильтрация по количеству рёбер ---
        print(f"\n3. Фильтрация по количеству рёбер (должно быть {len(ref_path) - 1})")
        filtered_paths = [p for p in all_map_paths if len(p['path']) == len(ref_path)]

        if not filtered_paths:
            print("   -> Не найдено маршрутов с совпадающим количеством рёбер. Сопоставление невозможно.")
            return None, None, debug_info
        print(f"   -> Осталось {len(filtered_paths)} кандидатов.")
        debug_info["filtered_paths"] = filtered_paths

        # --- Шаг 4: Вычисление критериев для TOPSIS ---
        print("\n4. Вычисление критериев для TOPSIS...")
        candidates = []
        criteria_list = []
        for path_info in filtered_paths:
            # Критерий 1: Общее соотношение длин
            crit1_len_ratio = path_info['length'] / ref_length if ref_length > 0 else 0

            # Критерий 2: Среднее отклонение пропорций рёбер
            map_p, ref_p = path_info['path'], ref_path
            edge_diffs = []
            for k in range(len(map_p) - 1):
                map_edge_len = self.map_distances[(map_p[k], map_p[k + 1])]
                ref_edge_len = self.photo_distances[(ref_p[k], ref_p[k + 1])]
                # Защита от деления на ноль
                edge_ratio = map_edge_len / ref_edge_len if ref_edge_len > 0 else 0
                edge_diffs.append(abs(crit1_len_ratio - edge_ratio))

            crit2_shape_similarity = np.mean(edge_diffs) if edge_diffs else 0

            candidates.append(path_info['endpoint_idx'])
            criteria_list.append([crit1_len_ratio, crit2_shape_similarity])
            print(
                f"   -> Кандидат {path_info['endpoint_idx']}: Критерии [{crit1_len_ratio:.3f}, {crit2_shape_similarity:.3f}]")

        debug_info["candidates"] = candidates
        debug_info["criteria_list"] = np.array(criteria_list)

        # --- Шаг 5: Применение TOPSIS ---
        best_match_idx = run_topsis(candidates, np.array(criteria_list))
        debug_info["best_match_idx"] = best_match_idx

        return self.map_points[best_match_idx], best_match_idx, debug_info


def visualize_results(matcher: GeoMatcher, ref_path: list, filtered_paths: list,
                      criteria_list: np.ndarray, candidates: list, best_match_idx: int):
    """
    Создает визуализацию процесса сопоставления.
    """
    if not filtered_paths:
        print("Нет данных для визуализации.")
        return

    # --- График 1: Карта сопоставления ---
    fig, ax1 = plt.subplots(1, 1, figsize=(10, 8))
    fig.suptitle("Визуализация процесса сопоставления", fontsize=16)

    # Отображаем все точки карты
    ax1.scatter(matcher.map_points[:, 0], matcher.map_points[:, 1], c='gray', alpha=0.6, label='Точки на карте')
    for i, p in enumerate(matcher.map_points):
        ax1.text(p[0], p[1] + 10, str(i), c='gray', fontsize=9)

    # Отображаем пути-кандидаты
    for i, path_info in enumerate(filtered_paths):
        path_points = matcher.map_points[path_info['path']]
        ax1.plot(path_points[:, 0], path_points[:, 1], marker='o', linestyle='--',
                 alpha=0.4,
                 label=f"Кандидат {path_info['endpoint_idx']}" if i < 5 else "")  # Подписываем только первые 5

    # Выделяем лучший путь
    best_path_info = next(p for p in filtered_paths if p['endpoint_idx'] == best_match_idx)
    best_path_points = matcher.map_points[best_path_info['path']]
    ax1.plot(best_path_points[:, 0], best_path_points[:, 1], marker='o', linestyle='-',
             color='green', linewidth=3, label=f"Лучший путь (индекс {best_match_idx})")

    # Выделяем стартовую и конечную точки
    start_map_idx = best_path_info['path'][-1]
    ax1.scatter(matcher.map_points[start_map_idx, 0], matcher.map_points[start_map_idx, 1],
                s=200, c='blue', marker='X', label='Стартовая точка (якорь)')
    ax1.scatter(matcher.map_points[best_match_idx, 0], matcher.map_points[best_match_idx, 1],
                s=200, c='red', marker='*', label='Найденная цель')

    ax1.set_title("Карта сопоставления путей")
    ax1.legend()
    ax1.grid(True, linestyle='--', alpha=0.5)
    ax1.set_aspect('equal', adjustable='box')

    # --- График 2: Пространство решений TOPSIS ---
    # Создаем новый график, чтобы избежать наложения
    fig2, ax2 = plt.subplots(1, 1, figsize=(8, 6))

    # Нормализуем критерии для TOPSIS (отклонение от идеала)
    processed_criteria = criteria_list.copy()
    processed_criteria[:, 0] = np.abs(processed_criteria[:, 0] - 1)

    # Отображаем всех кандидатов
    ax2.scatter(processed_criteria[:, 0], processed_criteria[:, 1], c='orange', alpha=0.8, label='Кандидаты')
    for i, txt in enumerate(candidates):
        ax2.annotate(txt, (processed_criteria[i, 0], processed_criteria[i, 1]))

    # Выделяем лучшее решение
    best_candidate_plot_idx = candidates.index(best_match_idx)
    ax2.scatter(processed_criteria[best_candidate_plot_idx, 0], processed_criteria[best_candidate_plot_idx, 1],
                s=150, c='green', marker='*', label=f'Лучший кандидат ({best_match_idx})')

    # Идеальная точка
    ax2.scatter(0, 0, s=200, c='red', marker='X', label='Идеальное решение')

    ax2.set_title("Пространство решений TOPSIS")
    ax2.set_xlabel("Критерий 1: |Соотношение длин - 1| (чем ближе к 0, тем лучше)")
    ax2.set_ylabel("Критерий 2: Схожесть формы (чем ближе к 0, тем лучше)")
    ax2.legend()
    ax2.grid(True, linestyle='--', alpha=0.5)
    ax2.axhline(0, color='black', linewidth=0.5)
    ax2.axvline(0, color='black', linewidth=0.5)

    plt.tight_layout()
    plt.show()


def visualize_initial_objects(map_coords: np.ndarray, photo_coords: np.ndarray,
                              map_points: np.ndarray, photo_points: np.ndarray,
                              best_match_coords: np.ndarray = None, best_match_index: int = None,
                              start_map_coords: tuple = None, target_photo_coords: tuple = None,
                              common_photo_point_idx: int = None):  # Добавляем common_photo_point_idx
    """
    Визуализирует все исходные объекты (прямоугольники) на карте и фото,
    а также центры и найденные соответствия.
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))
    fig.suptitle("Исходные объекты и их центры", fontsize=16)

    # --- Подграфик 1: Объекты на карте ---
    ax1.set_title("Объекты на карте")
    ax1.set_xlabel("X")
    ax1.set_ylabel("Y")
    ax1.set_aspect('equal', adjustable='box')
    ax1.grid(True, linestyle='--', alpha=0.6)

    min_x_map, max_x_map = np.min(map_coords[:, [0, 2]]), np.max(map_coords[:, [0, 2]])
    min_y_map, max_y_map = np.min(map_coords[:, [1, 3]]), np.max(map_coords[:, [1, 3]])
    ax1.set_xlim(min_x_map - 50, max_x_map + 50)
    ax1.set_ylim(min_y_map - 50, max_y_map + 50)

    for i, obj_coords in enumerate(map_coords):
        x1, y1, x2, y2 = obj_coords
        rect = patches.Rectangle((x1, y1), x2 - x1, y2 - y1, linewidth=1, edgecolor='blue', facecolor='lightblue',
                                 alpha=0.5, label=f'Объект {i}' if i == 0 else "")
        ax1.add_patch(rect)
        ax1.text(map_points[i, 0], map_points[i, 1] + 20, f'M{i}', color='darkblue', fontsize=9, ha='center')
        ax1.scatter(map_points[i, 0], map_points[i, 1], color='blue', s=20, zorder=5)  # Центр

    # Отмечаем стартовую точку на карте
    if start_map_coords is not None:
        start_map_idx = None
        for i, center in enumerate(map_points):
            if np.allclose(center, start_map_coords):
                start_map_idx = i
                break
        if start_map_idx is not None:
            ax1.scatter(map_points[start_map_idx, 0], map_points[start_map_idx, 1], color='cyan', s=200, marker='X',
                        zorder=10, label='Якорь на карте')
            ax1.text(map_points[start_map_idx, 0], map_points[start_map_idx, 1] - 30, f'Якорь M{start_map_idx}',
                     color='cyan', fontsize=10, ha='center')

    # Отмечаем найденное соответствие на карте
    if best_match_index is not None and best_match_coords is not None:
        ax1.scatter(best_match_coords[0], best_match_coords[1], color='red', s=250, marker='*', zorder=10,
                    label='Найденное соответствие')
        ax1.text(best_match_coords[0], best_match_coords[1] - 30, f'Цель M{best_match_index}', color='red', fontsize=10,
                 ha='center')

    # --- Подграфик 2: Объекты на фото ---
    ax2.set_title("Объекты на фото")
    ax2.set_xlabel("X")
    ax2.set_ylabel("Y")
    ax2.set_aspect('equal', adjustable='box')
    ax2.grid(True, linestyle='--', alpha=0.6)

    min_x_photo, max_x_photo = np.min(photo_coords[:, [0, 2]]), np.max(photo_coords[:, [0, 2]])
    min_y_photo, max_y_photo = np.min(photo_coords[:, [1, 3]]), np.max(photo_coords[:, [1, 3]])
    ax2.set_xlim(min_x_photo - 50, max_x_photo + 50)
    ax2.set_ylim(min_y_photo - 50, max_y_photo + 50)

    for i, obj_coords in enumerate(photo_coords):
        x1, y1, x2, y2 = obj_coords
        rect = patches.Rectangle((x1, y1), x2 - x1, y2 - y1, linewidth=1, edgecolor='green', facecolor='lightgreen',
                                 alpha=0.5, label=f'Объект {i}' if i == 0 else "")
        ax2.add_patch(rect)
        ax2.text(photo_points[i, 0], photo_points[i, 1] + 20, f'P{i}', color='darkgreen', fontsize=9, ha='center')
        ax2.scatter(photo_points[i, 0], photo_points[i, 1], color='green', s=20, zorder=5)  # Центр

    # Отмечаем стартовую точку на фото (якорь)
    if common_photo_point_idx is not None:
        ax2.scatter(photo_points[common_photo_point_idx, 0], photo_points[common_photo_point_idx, 1],
                    color='cyan', s=200, marker='X', zorder=10, label='Якорь на фото')
        ax2.text(photo_points[common_photo_point_idx, 0], photo_points[common_photo_point_idx, 1] - 30,
                 f'Якорь P{common_photo_point_idx}', color='cyan', fontsize=10, ha='center')

    # Отмечаем искомую цель на фото
    if target_photo_coords is not None:
        target_photo_idx_for_viz = None
        for i, center in enumerate(photo_points):
            if np.allclose(center, target_photo_coords):
                target_photo_idx_for_viz = i
                break
        if target_photo_idx_for_viz is not None:
            ax2.scatter(photo_points[target_photo_idx_for_viz, 0], photo_points[target_photo_idx_for_viz, 1],
                        color='red', s=250, marker='o', zorder=10, label='Искомая цель на фото')
            ax2.text(photo_points[target_photo_idx_for_viz, 0], photo_points[target_photo_idx_for_viz, 1] - 30,
                     f'Цель P{target_photo_idx_for_viz}', color='red', fontsize=10, ha='center')

    # Легенды на каждом графике
    handles1, labels1 = ax1.get_legend_handles_labels()
    by_label1 = dict(zip(labels1, handles1))
    ax1.legend(by_label1.values(), by_label1.keys())

    handles2, labels2 = ax2.get_legend_handles_labels()
    by_label2 = dict(zip(labels2, handles2))
    ax2.legend(by_label2.values(), by_label2.keys())

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])  # Adjust layout to prevent title overlap
    plt.show()


if __name__ == "__main__":
    # Координаты ключевых объектов на карте [x1, y1, x2, y2]
    map_coordinates = np.array([
        [386.5642395019531, 883.4074096679688, 611.2627563476562, 1044.19189453125],  # 0
        [702.1062622070312, 852.6519775390625, 937.92626953125, 1077.2757568359375],  # 1
        [758.607666015625, 1121.2928466796875, 1037.13037109375, 1229.0],  # 2
        [548.1726684570312, 234.66624450683594, 901.1254272460938, 510.3911437988281],  # 3
        [1124.4635009765625, 237.78834533691406, 1412.5557861328125, 514.2573852539062],  # 4
        [628.1311645507812, 622.7036743164062, 920.1611328125, 836.0609130859375],  # 5
        [264.75439453125, 620.1845703125, 445.0696716308594, 737.422607421875],  # 6
        [1142.1619873046875, 836.8211059570312, 1345.785400390625, 965.37353515625],  # 7
        [438.7745056152344, 584.6161809375, 607.2764892578125, 687.2083740234375],  # 8
        [454.1809997558594, 675.0203857421875, 646.0032348632812, 807.01171875],  # 9
        [40.660804748535156, 431.5293884277344, 251.85366821289062, 675.2212524414062],  # 10
    ])

    # Координаты объектов, распознанных на фото
    photo_coordinates = np.array([
        [66.16741180419922, 346.8585510253906, 190.1370391845703, 425.72174072265625],  # 0
        [14.484185218811035, 206.9029998779297, 96.39734649658203, 274.1340026855469],  # 1
        [459.77099609375, 321.11602783203125, 561.203857421875, 389.00732421875],  # 2
        [102.1821060180664, 238.49578857421875, 201.9081573486328, 302.3521728515625],  # 3
        [217.5886993408203, 335.67950439453125, 347.99749755859375, 430.6532897949219],  # 4
    ])

    matcher = GeoMatcher(map_coordinates, photo_coordinates)
    map_centers = matcher.map_points
    photo_centers = matcher.photo_points

    common_map_point_idx = 0
    common_photo_point_idx = 0
    start_point_coords = tuple(map_centers[common_map_point_idx])

    target_photo_idx_in_photo = 2
    target_photo_coords = tuple(photo_centers[target_photo_idx_in_photo])

    print(f"Общая стартовая точка (якорь) на карте (центр M{common_map_point_idx}): {start_point_coords}")
    print(f"Искомый объект на фото (центр P{target_photo_idx_in_photo}): {target_photo_coords}")

    # --- НОВАЯ ЧАСТЬ: Визуализация исходных объектов ---
    print("\nВизуализация исходных объектов и точек...")
    visualize_initial_objects(
        map_coords=map_coordinates,
        photo_coords=photo_coordinates,
        map_points=map_centers,
        photo_points=photo_centers,
        start_map_coords=start_point_coords,
        target_photo_coords=target_photo_coords,
        common_photo_point_idx=common_photo_point_idx
    )
    # --- КОНЕЦ НОВОЙ ЧАСТИ ---

    try:
        best_match_coords, best_match_index, debug_data = matcher.find_best_match(
            start_point_coords,
            target_photo_coords,
            known_start_photo_idx=common_photo_point_idx
        )

        if best_match_index is not None and best_match_coords is not None:
            print("\n" + "=" * 50)
            print("РЕЗУЛЬТАТ:")
            print(f"✅ Наилучшее соответствие для искомого объекта найдено на карте!")
            print(f"   Индекс объекта на карте: {best_match_index}")
            print(f"   Координаты центра: {tuple(np.round(best_match_coords, 2))}")
            print("=" * 50)

            visualize_results(
                matcher=matcher,
                ref_path=debug_data["ref_path"],
                filtered_paths=debug_data["filtered_paths"],
                criteria_list=debug_data["criteria_list"],
                candidates=debug_data["candidates"],
                best_match_idx=best_match_index
            )

        else:
            print("\n" + "=" * 50)
            print("❌ Не удалось найти однозначное соответствие.")
            print("=" * 50)

    except (ValueError, IndexError) as e:
        print(f"\n❌ ОШИБКА В ПРОЦЕССЕ ВЫПОЛНЕНИЯ: {e}")