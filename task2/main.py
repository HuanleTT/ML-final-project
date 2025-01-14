import numpy as np
import torch
import torch.nn as nn
import random

class PackingNet(nn.Module):

    def __init__(self, input_dim, hidden_dim, output_dim):
        super(PackingNet, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x

class Packing3D:
    def __init__(self, box_dim, num_items):
        self.box_dim = box_dim
        self.items = self.generate_items_by_cutting(box_dim, num_items)
        self.placed_items = []  
        self.box_volume = box_dim[0] * box_dim[1] * box_dim[2]
        self.net = PackingNet(input_dim=6, hidden_dim=64, output_dim=3)  
        self.optimizer = torch.optim.Adam(self.net.parameters(), lr=0.001)
        self.criterion = nn.MSELoss()

    def rotate(self, x, y, z):
        flag = random.randint(1, 666) % 6
        if flag == 0: return (x, y, z)
        elif flag == 1: return (x, z, y)
        elif flag == 2: return (y, x, z)
        elif flag == 3: return (y, z, x)
        elif flag == 4: return (z, x, y)
        else: return (z, y, x)

    def generate_items_by_cutting(self, box_dim, num_items): #使用algorithm1切割法生成items
        random.seed(102)
        items = []
        items.append(box_dim)
        while len(items) < num_items:
            item_tmp = random.choice(items)
            axis = random.choice(['length', 'width', 'height'])
            if axis == 'length' and item_tmp[0] > 1: 
                sp = random.randint(1, item_tmp[0] - 1)
                items.append(self.rotate(sp, item_tmp[1], item_tmp[2]))
                items.append(self.rotate(item_tmp[0] - sp, item_tmp[1], item_tmp[2]))
                items.remove(item_tmp)
            if axis == 'width' and item_tmp[1] > 1:
                sp = random.randint(1, item_tmp[1] - 1)
                items.append(self.rotate(item_tmp[0], sp, item_tmp[2]))
                items.append(self.rotate(item_tmp[0], item_tmp[1] - sp, item_tmp[2]))
                items.remove(item_tmp)
            if axis == 'height' and item_tmp[2] > 1:
                sp = random.randint(1, item_tmp[2] - 1)
                items.append(self.rotate(item_tmp[0], item_tmp[1], sp))
                items.append(self.rotate(item_tmp[0], item_tmp[1], item_tmp[2] - sp))
                items.remove(item_tmp)
        cnt = sum(item[0] * item[1] * item[2] for item in items)
        print(cnt)
        print(len(items))
        return items

    def is_valid_placement(self, pos, item_dim): #判断当前位置是否合法
        x, y, z = pos
        l, w, h = item_dim
        bx, by, bz = self.box_dim

        if x + l > bx or y + w > by or z + h > bz or x < 0 or y < 0 or z < 0:
            return False

        for placed in self.placed_items:
            px, py, pz, pl, pw, ph = placed
            if px >= x + l or px + pl <= x:
                continue
            elif py >= y + w or py + pw <= y:
                continue
            elif pz >= z + h or pz + ph <= z:
                continue
            else: return False

        return True

    def place_item(self, pos, item_dim):
        self.placed_items.append((*pos, *item_dim))

    def remove_item(self):
        if self.placed_items:
            self.placed_items.pop()

    def dfs(self, items, depth=0):
        if not items:
            return  

        item = items[0]
        remaining_items = items[1:]
        
        item_tensor = torch.tensor(item + self.box_dim, dtype=torch.float32)
        predicted_pos = self.net(item_tensor).detach().numpy()
        predicted_pos = tuple(int(round(coord)) for coord in predicted_pos)

        if self.is_valid_placement(predicted_pos, item):
            self.place_item(predicted_pos, item)
            self.dfs(remaining_items, depth + 1)
            return
        
        for x in range(self.box_dim[0]):
            for y in range(self.box_dim[1]):
                for z in range(self.box_dim[2]):
                    pos = (x, y, z)
                    if self.is_valid_placement(pos, item):
                        self.place_item(pos, item)
                        self.dfs(remaining_items, depth + 1)  
                        return
    
    def train_network(self, epochs=100):
        data = []
        labels = []
        for item in self.items:
            for _ in range(10):  
                box_state = np.random.uniform(0, 1, size=3)  
                input_data = np.concatenate((item, box_state))
                label = np.random.uniform(0, 1, size=3)  
                data.append(input_data)
                labels.append(label)

        data = torch.tensor(data, dtype=torch.float32)
        labels = torch.tensor(labels, dtype=torch.float32)

        for epoch in range(epochs):
            self.optimizer.zero_grad()
            outputs = self.net(data)
            loss = self.criterion(outputs, labels)
            loss.backward()
            self.optimizer.step()
            if (epoch + 1) % 10 == 0:
                print(f"Epoch [{epoch + 1}/{epochs}], Loss: {loss.item():.4f}")
    
    def calculate_space_utilization(self):
        used_volume = sum(l * w * h for _, _, _, l, w, h in self.placed_items)
        return used_volume / self.box_volume

    def solve(self):
        self.dfs(self.items)
        placed_count = len(self.placed_items)
        space_utilization = self.calculate_space_utilization()
        return self.placed_items, placed_count, space_utilization

if __name__ == "__main__":
    box_dim = (100, 100, 100)

    num_items = 50

    packing = Packing3D(box_dim, num_items)

    packing.train_network(epochs=50)

    solution, placed_count, space_utilization = packing.solve()

    for item in solution:
        print(item)
    print(f"成功放入箱子里的物品个数: {placed_count}")
    print(f"空间利用率: {space_utilization:.2%}")