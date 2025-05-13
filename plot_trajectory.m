function plot_trajectory(estPoseFile, gtPoseFile, outputVideoFile)

    % === Load poses ===
    est_data = readmatrix(estPoseFile);    % Estimated trajectory
    gt_data  = readmatrix(gtPoseFile);  % Ground truth trajectory

    % === Number of frames ===
    N = min(size(est_data,1), size(gt_data,1));

    % === Preallocate ===
    x_est = zeros(N, 1); y_est = zeros(N, 1); z_est = zeros(N, 1); R_est = zeros(3, 3, N);
    x_gt  = zeros(N, 1); y_gt  = zeros(N, 1); z_gt  = zeros(N, 1);  R_gt = zeros(3, 3, N);

    % === Convert poses ===
    for i = 1:N
        % Estimated
        T = reshape(est_data(i, :), [4, 3])';
        R_kitti = T(:, 1:3);
        t_kitti = T(:, 4);

        x_est(i) =  t_kitti(1);
        y_est(i) = -t_kitti(2);
        z_est(i) =  t_kitti(3);
        R_est(:,:,i) = [ R_kitti(:,1), R_kitti(:,2), R_kitti(:,3) ];

        % Ground truth
        T = reshape(gt_data(i, :), [4, 3])';
        R_kitti = T(:, 1:3);
        t_kitti = T(:, 4);

        x_gt(i) =  t_kitti(1);
        y_gt(i) = t_kitti(2);
        z_gt(i) =  t_kitti(3);
        R_gt(:,:,i) = [ R_kitti(:,1), R_kitti(:,2), R_kitti(:,3) ];

    end

    % === Camera model ===
    axisLength = 1.0;
    camVertices = axisLength * [
        -0.5 -0.5  1;
         0.5 -0.5  1;
         0.5  0.5  1;
        -0.5  0.5  1;
         0.0  0.0  0;
    ];

    % === Video setup ===
    videoFileName = outputVideoFile;
    v = VideoWriter(videoFileName, 'Uncompressed AVI');
    v.FrameRate = 30;
    open(v);

    % === Create figure ===
    fig = figure;
    set(fig, 'Position', [100, 100, 1200, 900]);
    set(gca, 'Units', 'normalized', 'Position', [0 0 1 1]);
    hold on;
    rotate3d on;

    % Plot trajectories
    hEst = plot3(x_est, y_est, z_est, 'r-',  'LineWidth', 2);
    hGT  = plot3(x_gt,  y_gt,  z_gt,  'b--', 'LineWidth', 2);

    legend([hEst, hGT], ...
           {'Estimated', 'GroundTruth'}, ...
           'Location','best');

    xlabel('X (left/right)');
    ylabel('Y (up/down)');
    zlabel('Z (forward)');
    title('KITTI Camera Trajectory Comparison');

    xmin = floor(min([x_est; x_gt]));
    xmax = ceil( max([x_est; x_gt]));
    ymin = floor(min([y_est; y_gt]));
    ymax = ceil( max([y_est; y_gt]));
    zmin = floor(min([z_est; z_gt]));
    zmax = ceil( max([z_est; z_gt]));

    range_max = max([xmax - xmin, ymax - ymin, zmax - zmin]);
    center = [mean([xmin, xmax]), mean([ymin, ymax]), mean([zmin, zmax])];
    half_range = range_max / 2;

    xlim([center(1) - half_range, center(1) + half_range]);
    ylim([center(2) - half_range, center(2) + half_range]);
    zlim([center(3) - half_range, center(3) + half_range]);
    axis equal; axis vis3d;

    view(3);

    % Camera patches
    set(camPatchEst,'HandleVisibility','off')
    set(camPatchGT, 'HandleVisibility','off')

    % First frame
    frameSize = [600, 800];
    frame = getframe(fig);
    frameImage(:,:,:) = imresize(frame.cdata, frameSize);
    writeVideo(v, frameImage);

    startAz = 45;
    startEl = 30;
    endAz   = startAz + 360;
    numFrames = ceil((N/10));
    azimuths = linspace(startAz, endAz, numFrames);

    frameIdx = 0;
    for i = 1:10:N
        frameIdx = frameIdx + 1;

        R = R_est(:,:,i);
        vtx = (R * camVertices')' + [x_est(i), y_est(i), z_est(i)];
        set(camPatchEst, 'Vertices', vtx);

        R = R_gt(:,:,i);
        vtx = (R * camVertices')' + [x_gt(i), y_gt(i), z_gt(i)];
        set(camPatchGT, 'Vertices', vtx);

        az = azimuths(frameIdx);
        view(az, startEl);

        drawnow;

        frame = getframe(fig);
        frameImage = imresize(frame.cdata, frameSize);
        writeVideo(v, frameImage);
    end

    close(v);
end
