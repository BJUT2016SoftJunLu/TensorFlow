import jieba


# 使用用户字典
jieba.load_userdict('D:\\ProgramData\\Anaconda3\\Lib\\site-packages\\jieba\\user_dict.txt')

def get_text():
    text = []
    wp_file = open("wp_file.csv", 'w', encoding='utf_8_sig')
    with open("atec_nlp_sim_train_add.csv", 'r', encoding='utf_8_sig') as file:
        line_nums = 0
        for line in file:
            line_nums += 1
            if line_nums == 100:
                break
            sample_list = line.split("\t")
            wp_first = jieba.cut(sample_list[1])
            wp_second = jieba.cut(sample_list[2])
            text.append(" ".join(wp_first).split(" "));text.append(" ".join(wp_second).split(" "))
            sample_wp = sample_list[0] + "\t" + "|".join(wp_first) + "\t" + "|".join(wp_second) + "\t" + sample_list[3]
            wp_file.write(sample_wp)
    wp_file.close()
    return text

def main():
    print(get_text())

if __name__ == '__main__':
    main()
