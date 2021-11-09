from autocorrection.correct import AutoCorrection 
if __name__ == '__main__':
    sent = "20 Cong nhận su dụng dt chems cheets giams đốc cũ. Bùng nỗ nanj troomj cướp caay cảnh 20/10/2000"
    correction = AutoCorrection(model_name = "SoftmaskedBert")
    results = correction.correct(sent)
    print("Before:", sent)
    print("After:", results)
