function data_trva()
    seed = 1234567890;
    rng(seed);

    name = 'data';

    inits_tr = [-2,0.5; 2,-0.5; -0.3,-0.3; 0.3,0.3];
    inits_va = [-1.5,0; 1.5,0; -0.5,0.5; 0.5,-0.5];
    
    len_epi = 20; tmax = 1.5;
    inits = [inits_tr; inits_va];
    num_epi = size(inits,1);
    tr = 1:size(inits_tr,1);
    va = tr(end)+1:tr(end)+size(inits_va,1);

    T_all = zeros(num_epi, len_epi);
    Y_all = zeros(num_epi, len_epi, 2);
    dotY_all = zeros(num_epi, len_epi, 2);
    for i=1:num_epi
        [T_, Y_, dotY_] = circsurf(len_epi, inits(i,:), tmax);
        T_all(i,:) = T_;
        Y_all(i,:,:) = Y_;
        dotY_all(i,:,:) = dotY_;
    end

    noisestd = 0;

    Y_all = Y_all + randn(size(Y_all))*noisestd;
    dotY_all = dotY_all + randn(size(dotY_all))*noisestd;

    % plot
    if true
        figure;
        hold on;
        for i=tr
            plot(Y_all(i,:,1), Y_all(i,:,2), 'b');
            plot(inits(i,1), inits(i,2), 'bo')
        end
        for i=va
            plot(Y_all(i,:,1), Y_all(i,:,2), 'r');
            plot(inits(i,1), inits(i,2), 'ro')
        end
        plot(cos(linspace(0,2*pi,100)), sin(linspace(0,2*pi,100)), 'g');
        hold off;
    end
    
    % save
    if true
        T = T_all(tr,:);
        Y = Y_all(tr,:,:);
        dotY = dotY_all(tr,:,:);
        save(sprintf('./%s_tr.mat',name), 'T', 'Y', 'dotY');

        T = T_all(va,:);
        Y = Y_all(va,:,:);
        dotY = dotY_all(va,:,:);
        save(sprintf('./%s_va.mat',name), 'T', 'Y', 'dotY');

        clear T Y dotY T_all Y_all dotY_all;
        save(sprintf('./%s_info.mat',name));
        
        fprintf('data saved\n');
    end
end
