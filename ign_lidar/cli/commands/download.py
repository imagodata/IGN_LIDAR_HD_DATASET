"""Download command for IGN LiDAR HD CLI."""

import logging
import sys
from pathlib import Path

import click

logger = logging.getLogger(__name__)


@click.command()
@click.argument('output_dir', type=click.Path())
@click.option('--bbox', '-b', type=float, nargs=4, metavar='XMIN YMIN XMAX YMAX',
              help='Bounding box coordinates (Lambert93)')
@click.option('--position', '-p', type=float, nargs=2, metavar='X Y',
              help='Download tiles around position (Lambert93 coordinates)')
@click.option('--radius', '-r', default=5000, type=float,
              help='Radius around position in meters (default: 5000)')
@click.option('--max-concurrent', '-j', default=3, type=int,
              help='Maximum concurrent downloads (default: 3)')
@click.option('--force', '-f', is_flag=True,
              help='Force download even if files already exist')
@click.option('--list-locations', is_flag=True,
              help='List available strategic locations')
@click.option('--location', '-l', type=str,
              help='Download by strategic location name')
@click.pass_context
def download_command(ctx, output_dir, bbox, position, radius, max_concurrent, force, list_locations, location):
    """Download IGN LiDAR HD tiles by bounding box, position, or location."""
    try:
        from ...downloader import IGNLiDARDownloader
        from ...datasets.strategic_locations import STRATEGIC_LOCATIONS
    except ImportError as e:
        logger.error(f"Missing dependencies for download: {e}")
        raise click.ClickException("Download functionality requires additional dependencies")
    
    verbose = ctx.obj.get('verbose', False) if ctx.obj else False
    
    if list_locations:
        click.echo("Available strategic locations:")
        click.echo("=" * 40)
        for name, data in STRATEGIC_LOCATIONS.items():
            click.echo(f"  {name:20s} - {data.get('description', 'No description')}")
        return
    
    try:
        downloader = IGNLiDARDownloader(Path(output_dir), max_concurrent=max_concurrent)
        
        if verbose:
            logger.info(f"Downloading to: {output_dir}")
            logger.info(f"Max concurrent: {max_concurrent}")
        
        tiles_to_download = []

        if location:
            if location not in STRATEGIC_LOCATIONS:
                raise click.ClickException(f"Unknown location: {location}")
            loc_data = STRATEGIC_LOCATIONS[location]
            bbox = list(loc_data['bbox'])  # (xmin, ymin, xmax, ymax) WGS84
            click.echo(f"Using location '{location}' ({loc_data.get('category', '?')}): bbox={bbox}")

        if bbox:
            # Interroge le WFS IGN (qui parle WGS84 nativement) plutôt que la
            # conversion locale Lambert93 qui ne marche pas avec des degrés.
            features = downloader.fetch_available_tiles(bbox=tuple(bbox)).get("features", [])
            tiles_to_download = [
                f["properties"]["name"]
                for f in features
                if f.get("properties", {}).get("name")
            ]
        elif position:
            x, y = position
            tile_infos = downloader.find_tiles_by_position(x, y, radius_km=radius / 1000.0)
            tiles_to_download = [t["filename"] for t in tile_infos]
        else:
            raise click.ClickException("Must specify --bbox, --position, or --location")
        
        if not tiles_to_download:
            click.echo("No tiles found for the specified area")
            return
            
        click.echo(f"Found {len(tiles_to_download)} tiles to download")
        
        if verbose:
            for i, tile in enumerate(tiles_to_download[:5]):  # Show first 5
                click.echo(f"  {i+1}. {tile}")
            if len(tiles_to_download) > 5:
                click.echo(f"  ... and {len(tiles_to_download) - 5} more")
        
        # Download tiles
        results = downloader.batch_download(
            tiles_to_download,
            skip_existing=not force
        )
        
        success = [tile for tile, status in results.items() if status]
        failed = [tile for tile, status in results.items() if not status]
        
        click.echo(f"\n✅ Download complete!")
        click.echo(f"  Successfully downloaded: {len(success)}")
        if failed:
            click.echo(f"  Failed: {len(failed)}")
            for tile in failed:
                click.echo(f"    - {tile}")
    
    except Exception as e:
        click.echo(f"❌ Download error: {e}", err=True)
        sys.exit(1)