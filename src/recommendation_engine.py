def generate_recommendation(employee, prob, threshold=0.4):
    recommendations = []

    if prob >= threshold:
        if employee['OverTime'] == 1:
            recommendations.append("Reduce overtime workload")

        if employee['MonthlyIncome'] < 4000:
            recommendations.append("Consider salary increase")

        if employee['JobSatisfaction'] <= 2:
            recommendations.append("Improve job satisfaction")

        if employee['WorkLifeBalance'] <= 2:
            recommendations.append("Enhance work-life balance")

    else:
        if employee['JobSatisfaction'] <= 2:
            recommendations.append("Monitor job satisfaction")

    return recommendations