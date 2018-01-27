function answer=Normalize(att)
    minVal = min(att);
    maxVal = max(att);
    answer = (att - minVal) / ( maxVal - minVal );