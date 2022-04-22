function num = number_poly_terms(degree, N)
    num = 0;
    for d = 0: degree
        num = num + nchoosek(d + N - 1, N - 1);
    end
end
