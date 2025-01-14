import numpy as np
import csv
import logging

logging.basicConfig(filename='debug_output2.log', level=logging.DEBUG, 
                    format='%(asctime)s - %(levelname)s - %(message)s')

# 容器尺寸
container_sizes = [
    (35, 23, 13), 
    (37, 26, 13), 
    (38, 26, 13), 
    (40, 28, 16), 
    (42, 30, 18), 
    (42, 30, 40), 
    (52, 40, 17), 
    (54, 45, 36)
]

# 判断物品是否可以放入容器
def can_fit(container, item, position):
    """
    判断物品是否可以放入容器中，考虑物品尺寸与容器尺寸，以及当前位置
    """
    container_length, container_width, container_height = container
    item_length, item_width, item_height = item

    # 物品放置后是否超出容器的尺寸范围
    x, y, z = position
    if x + item_length > container_length or y + item_width > container_width or z + item_height > container_height:
        return False
    return True

# 判断物品是否与已经放置的物品发生重叠
def is_overlap(position, item_rotation, placed_item):
    """
    检查当前物品与已放置物品是否重叠
    """
    placed_item_rotation, placed_position, _, _ = placed_item
    placed_x, placed_y, placed_z = placed_position
    
    item_length, item_width, item_height = item_rotation
    placed_length, placed_width, placed_height = placed_item_rotation
    
    # 检查物品是否在放置区域发生重叠
    if (position[0] + item_length > placed_x and position[0] < placed_x + placed_length and
        position[1] + item_width > placed_y and position[1] < placed_y + placed_width and
        position[2] + item_height > placed_z and position[2] < placed_z + placed_height):
        return True
    return False

# 获取物品放置的合法位置
def find_position_for_item(container, item, placed_items):
    """
    找到容器中可以放置物品的位置，考虑所有可能的旋转方式
    """
    container_length, container_width, container_height = container
    item_length, item_width, item_height = item

    # 尝试每个旋转后的物品
    for rotation in [
        (item_length, item_width, item_height),
        (item_length, item_height, item_width),
        (item_width, item_length, item_height),
        (item_width, item_height, item_length),
        (item_height, item_length, item_width),
        (item_height, item_width, item_length)
    ]:
        # 在容器中尝试每个位置
        for x in range(int(container_length - rotation[0] + 1)):  # 转换为整数
            for y in range(int(container_width - rotation[1] + 1)):  # 转换为整数
                for z in range(int(container_height - rotation[2] + 1)):  # 转换为整数
                    position = (x, y, z)

                    # 检查该位置是否与已放置的物品发生冲突
                    if not any(is_overlap(position, rotation, placed_item) for placed_item in placed_items):
                        # 额外检查：确保物品可以放入容器（即物品不超出容器边界）
                        if can_fit(container, rotation, position):
                            logging.debug(f"Item placed at position {position} with rotation {rotation}")
                            return position, rotation
    return None, None

# 打包算法 - First Fit Decreasing (FFD)
# 打包算法 - First Fit Decreasing (FFD)
def first_fit_decreasing(items, container_sizes):
    # 按物品体积降序排列
    items = sorted(items, key=lambda x: x[0] * x[1] * x[2], reverse=True)
    
    containers = []  # 存放容器的列表
    containers_placed_items = []  # 存放每个容器内已放置物品的列表
    placed_item_ids = set()

    # 遍历每个物品
    for item_id, item in enumerate(items):
        placed = False
        # 尝试放入已有的容器
        for container_idx, (container, placed_items) in enumerate(zip(containers, containers_placed_items)):
            for rotation in [
                (item[0], item[1], item[2]),
                (item[0], item[2], item[1]),
                (item[1], item[0], item[2]),
                (item[1], item[2], item[0]),
                (item[2], item[0], item[1]),
                (item[2], item[1], item[0])
            ]:
                # 检查该旋转是否已被放置过
                if (item_id, rotation) in placed_item_ids:
                    continue  # 如果该旋转形式已经被放置，跳过
                
                # 查找放置位置
                position, rotated_item = find_position_for_item(container, item, placed_items)  
                if position is not None:  # 如果找到了合法的位置
                    placed_item_ids.add((item_id, rotation))  # 标记该旋转形式已经放置
                    placed_items.append((rotated_item, position, True, container_idx))  # 物品旋转并放置
                    placed = True
                    break  # 物品放入容器后跳出旋转循环
            if placed:
                break  # 物品放入容器后跳出容器循环

        # 如果物品不能放入任何已有容器
        if not placed:
            # 尝试所有容器，直到找到放得下的位置
            for container_idx, container in enumerate(container_sizes):
                for rotation in [
                    (item[0], item[1], item[2]),
                    (item[0], item[2], item[1]),
                    (item[1], item[0], item[2]),
                    (item[1], item[2], item[0]),
                    (item[2], item[0], item[1]),
                    (item[2], item[1], item[0])
                ]:
                    # 检查该旋转是否已被放置过
                    if (item_id, rotation) in placed_item_ids:
                        continue  # 如果该旋转形式已经被放置，跳过
                    
                    # 查找放置位置
                    position, rotated_item = find_position_for_item(container, item, [])
                    if position is not None:  # 如果找到了合法的位置
                        placed_item_ids.add((item_id, rotation))  # 标记该旋转形式已经放置
                        containers.append(container)  # 新增一个容器
                        containers_placed_items.append([(rotated_item, position, True, len(containers) - 1)])  # 新增容器中的物品
                        placed = True
                        break  # 找到位置后跳出旋转循环
                if placed:
                    break  # 找到放置位置后跳出容器循环

        # 如果物品依然无法放置，创建一个新的容器并放置
        if not placed:
            logging.warning(f"Warning: No valid position for item {item} in any container. Skipping item.")
    
    return containers, containers_placed_items



# items = [
#     (28.4, 20.2, 12),  # 长, 宽, 高
#     (35.6, 28.8, 12.6),
#     (37.4, 24.8, 13.8),
#     (52.1, 47.0, 5.3),
#     (38.6, 24.0, 13.8)
# ]

def read_items_from_csv(file_path):
    items = []
    with open(file_path, mode='r', encoding='utf-8') as file:
        csv_reader = csv.reader(file)
        for row in csv_reader:
            if len(row) == 4:  # 每行有4个数（长、宽、高、数量）
                try:
                    length, width, height, quantity = map(float, row)
                    quantity = int(row[3])  
                      # 分别获取长、宽、高和数量
                    # 根据数量将物品添加到列表中
                    items.extend([(length, width, height)] * quantity)
                except ValueError:
                    print(f"Warning: Invalid data found in row {row}. Skipping row.")
    return items

# 示例：假设文件路径为 'items.csv'
file_path = "C:/Users/王翔宇/downloads/task3.csv"  # 修改为实际的文件路径
items = read_items_from_csv(file_path)

containers, placed_items = first_fit_decreasing(items, container_sizes)

def calculate_total_volume(containers):
    total_volume = 0
    for container in containers:
        length, width, height = container
        volume = length * width * height
        total_volume += volume
    return total_volume
total_volume = calculate_total_volume(containers)
# 输出结果
logging.info(f"Containers used:%d", len(containers))
logging.info(f"Total volume of all containers: {total_volume} cubic units")